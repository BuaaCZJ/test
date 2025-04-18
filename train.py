import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.yolov11 import YOLOv11
from utils.datasets import create_dataloader
from utils.utils import load_classes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像大小')
    parser.add_argument('--weights', type=str, default='', help='初始权重路径')
    parser.add_argument('--device', default='', help='cuda设备, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--save-dir', type=str, default='runs/train', help='保存结果的目录')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    return parser.parse_args()

def train(model, dataloader, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs}')
    for i, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = model(imgs)
        
        # 计算损失
        loss = compute_loss(outputs, targets, model)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新进度条
        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (i + 1)})
    
    return total_loss / len(dataloader)

def compute_loss(outputs, targets, model):
    """计算YOLOv11损失函数"""
    # 这里简化了损失计算，实际应包括边界框回归损失、置信度损失和分类损失
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    
    # 定义损失函数
    BCEcls = nn.BCEWithLogitsLoss(reduction='mean')
    BCEobj = nn.BCEWithLogitsLoss(reduction='mean')
    MSEbox = nn.MSELoss(reduction='mean')
    
    # 处理每个输出层
    for i, output in enumerate(outputs):
        # 这里需要根据实际情况实现目标分配和损失计算
        # 简化版：假设已经有了目标分配结果
        b, anchor, grid_y, grid_x, _ = targets.shape  # 假设targets已经被处理成对应格式
        
        # 边界框回归损失
        lbox += MSEbox(output[..., :4], targets[..., :4])
        
        # 置信度损失
        lobj += BCEobj(output[..., 4], targets[..., 4])
        
        # 分类损失
        lcls += BCEcls(output[..., 5:], targets[..., 5:])
    
    # 总损失 = 边界框损失 + 置信度损失 + 分类损失
    loss = lbox + lobj + lcls
    
    return loss

def validate(model, dataloader, device):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc='Validating'):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(imgs)
            
            # 计算损失
            loss = compute_loss(outputs, targets, model)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.device:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device.isdigit() else 'cpu')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据配置
    with open(args.data, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # 获取类别名称
    class_names = load_classes(data_dict['names']) if 'names' in data_dict else None
    num_classes = len(class_names)
    
    # 创建模型
    model = YOLOv11(num_classes=num_classes)
    
    # 加载预训练权重
    if args.weights:
        if os.path.exists(args.weights):
            model.load_state_dict(torch.load(args.weights, map_location=device))
            print(f'Loaded weights from {args.weights}')
    
    model = model.to(device)
    
    # 创建数据加载器
    train_loader = create_dataloader(
        data_dict['train'], data_dict['train_labels'],
        img_size=args.img_size, batch_size=args.batch_size,
        augment=True, class_names=class_names
    )
    
    val_loader = create_dataloader(
        data_dict['val'], data_dict['val_labels'],
        img_size=args.img_size, batch_size=args.batch_size,
        augment=False, class_names=class_names
    )
    
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练
        train_loss = train(model, train_loader, optimizer, device, epoch, args.epochs)
        
        # 验证
        val_loss = validate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))
        
        # 保存最后一个模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'last.pt'))
        
        # 打印结果
        print(f'Epoch {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

if __name__ == '__main__':
    main()