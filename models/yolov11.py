import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """标准卷积块：Conv + BatchNorm + SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.conv1 = ConvBlock(channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBlock(hidden_channels, channels, kernel_size=3)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class CSPBlock(nn.Module):
    """CSP (Cross Stage Partial) 块"""
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels//2, kernel_size=1)
        self.conv2 = ConvBlock(in_channels, out_channels//2, kernel_size=1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels//2) for _ in range(num_blocks)])
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.blocks(y1)
        y2 = self.conv2(x)
        y = torch.cat([y1, y2], dim=1)
        return self.conv3(y)

class SPPFBlock(nn.Module):
    """空间金字塔池化块"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, kernel_size=1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = x
        y2 = self.m(x)
        y3 = self.m(y2)
        y4 = self.m(y3)
        y = torch.cat([y1, y2, y3, y4], dim=1)
        return self.conv2(y)

class YOLOv11(nn.Module):
    """YOLOv11模型实现"""
    def __init__(self, num_classes=80, input_channels=3):
        super().__init__()
        self.num_classes = num_classes
        
        # 主干网络
        self.stem = ConvBlock(input_channels, 64, kernel_size=3, stride=2)
        
        # 下采样和特征提取
        self.dark1 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2),
            CSPBlock(128, 128, num_blocks=3)
        )
        
        self.dark2 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2),
            CSPBlock(256, 256, num_blocks=6)
        )
        
        self.dark3 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=2),
            CSPBlock(512, 512, num_blocks=9)
        )
        
        self.dark4 = nn.Sequential(
            ConvBlock(512, 1024, kernel_size=3, stride=2),
            CSPBlock(1024, 1024, num_blocks=3),
            SPPFBlock(1024, 1024, kernel_size=5)
        )
        
        # 特征融合网络
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.lateral_conv1 = ConvBlock(1024, 512, kernel_size=1)
        self.fusion_conv1 = CSPBlock(1024, 512, num_blocks=3)
        
        self.lateral_conv2 = ConvBlock(512, 256, kernel_size=1)
        self.fusion_conv2 = CSPBlock(512, 256, num_blocks=3)
        
        self.lateral_conv3 = ConvBlock(256, 128, kernel_size=1)
        self.fusion_conv3 = CSPBlock(256, 128, num_blocks=3)
        
        # 预测头
        self.head_conv1 = ConvBlock(128, 256, kernel_size=3)
        self.head_conv2 = ConvBlock(256, 512, kernel_size=3)
        self.head_conv3 = ConvBlock(512, 1024, kernel_size=3)
        
        # 输出层 - 每个预测包含 (5 + num_classes) 个值：x, y, w, h, obj_conf, class_probs
        self.output1 = nn.Conv2d(256, 3 * (5 + num_classes), kernel_size=1)
        self.output2 = nn.Conv2d(512, 3 * (5 + num_classes), kernel_size=1)
        self.output3 = nn.Conv2d(1024, 3 * (5 + num_classes), kernel_size=1)
    
    def forward(self, x):
        # 主干网络
        x = self.stem(x)
        x = self.dark1(x)
        feat1 = self.dark2(x)  # P3
        feat2 = self.dark3(feat1)  # P4
        feat3 = self.dark4(feat2)  # P5
        
        # 特征融合 - 自顶向下路径
        p5 = self.lateral_conv1(feat3)
        p5_upsample = self.upsample(p5)
        p5_fusion = torch.cat([p5_upsample, feat2], dim=1)
        p4 = self.fusion_conv1(p5_fusion)
        
        p4 = self.lateral_conv2(p4)
        p4_upsample = self.upsample(p4)
        p4_fusion = torch.cat([p4_upsample, feat1], dim=1)
        p3 = self.fusion_conv2(p4_fusion)
        
        # 预测头
        p3_out = self.head_conv1(p3)
        out1 = self.output1(p3_out)  # 小目标
        
        p3_downsample = F.max_pool2d(p3_out, kernel_size=2, stride=2)
        p4_cat = torch.cat([p3_downsample, p4], dim=1)
        p4_out = self.head_conv2(p4_cat)
        out2 = self.output2(p4_out)  # 中目标
        
        p4_downsample = F.max_pool2d(p4_out, kernel_size=2, stride=2)
        p5_cat = torch.cat([p4_downsample, p5], dim=1)
        p5_out = self.head_conv3(p5_cat)
        out3 = self.output3(p5_out)  # 大目标
        
        return [out1, out2, out3]
    
    def _make_grid(self, nx, ny, anchor_vec):
        # 创建网格点和锚框
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        anchor_grid = anchor_vec.clone().view((1, -1, 1, 1, 2)).expand(1, -1, ny, nx, 2).float()
        return grid, anchor_grid