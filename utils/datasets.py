import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

class YOLODataset(Dataset):
    """YOLO数据集加载器"""
    def __init__(self, img_dir, label_dir, img_size=640, augment=True, class_names=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.class_names = class_names
        
        # 获取所有图像文件
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.img_files.sort()
        
        # 对应的标签文件
        self.label_files = [os.path.join(label_dir, os.path.splitext(os.path.basename(img_file))[0] + '.txt') 
                           for img_file in self.img_files]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        # 加载图像
        img_path = self.img_files[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 加载标签
        label_path = self.label_files[index]
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    class_id = int(data[0])
                    # YOLO格式：中心点x, 中心点y, 宽, 高（归一化到0-1）
                    x_center, y_center, width, height = map(float, data[1:5])
                    labels.append([class_id, x_center, y_center, width, height])
        
        labels = np.array(labels)
        
        # 数据增强
        if self.augment and len(labels) > 0:
            img, labels = self.augment_data(img, labels)
        
        # 调整图像大小
        img, labels = self.resize(img, labels)
        
        # 转换为张量
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).float() / 255.0  # 归一化到0-1
        
        # 创建目标张量
        targets = torch.zeros((len(labels), 6))  # [batch_idx, class, x, y, w, h]
        if len(labels):
            targets[:, 1:] = torch.from_numpy(labels)
        
        return img, targets
    
    def resize(self, img, labels):
        """调整图像大小并相应地调整标签"""
        h, w, _ = img.shape
        ratio = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        # 调整图像大小
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建新图像（填充黑色）
        new_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        # 将调整大小的图像放在中心
        new_img[(self.img_size - new_h) // 2:(self.img_size - new_h) // 2 + new_h, 
               (self.img_size - new_w) // 2:(self.img_size - new_w) // 2 + new_w, :] = resized_img
        
        # 调整标签
        if len(labels) > 0:
            # 调整中心点坐标和尺寸
            labels[:, 1] = labels[:, 1] * new_w / w + (self.img_size - new_w) / 2 / self.img_size
            labels[:, 2] = labels[:, 2] * new_h / h + (self.img_size - new_h) / 2 / self.img_size
            labels[:, 3] = labels[:, 3] * new_w / w / self.img_size
            labels[:, 4] = labels[:, 4] * new_h / h / self.img_size
        
        return new_img, labels
    
    def augment_data(self, img, labels):
        """数据增强：随机水平翻转、亮度、对比度和饱和度调整"""
        h, w, _ = img.shape
        
        # 随机水平翻转
        if random.random() < 0.5:
            img = cv2.flip(img, 1)  # 水平翻转
            if len(labels) > 0:
                labels[:, 1] = 1 - labels[:, 1]  # 翻转x坐标
        
        # 随机亮度、对比度和饱和度调整
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 亮度调整
        value_factor = random.uniform(0.8, 1.2)
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * value_factor, 0, 255).astype(np.uint8)
        
        # 对比度调整
        contrast_factor = random.uniform(0.8, 1.2)
        mean_val = np.mean(hsv_img[:, :, 2])
        hsv_img[:, :, 2] = np.clip((hsv_img[:, :, 2] - mean_val) * contrast_factor + mean_val, 0, 255).astype(np.uint8)
        
        # 饱和度调整
        saturation_factor = random.uniform(0.8, 1.2)
        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
        
        # 转回RGB
        img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        
        return img, labels

def create_dataloader(img_dir, label_dir, img_size=640, batch_size=16, augment=True, class_names=None):
    """创建数据加载器"""
    dataset = YOLODataset(img_dir, label_dir, img_size, augment, class_names)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return dataloader

def collate_fn(batch):
    """自定义批处理函数，处理不同大小的目标"""
    imgs, targets = zip(*batch)
    # 添加批次索引
    for i, boxes in enumerate(targets):
        boxes[:, 0] = i
    return torch.stack(imgs, 0), torch.cat(targets, 0)