import argparse
import os
import time
import cv2
import torch
import numpy as np
from PIL import Image

from models.yolov11 import YOLOv11
from utils.utils import non_max_suppression, xywh2xyxy, plot_boxes, load_classes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--source', type=str, default='data/images', help='图像源，可以是图像、视频或目录')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--img-size', type=int, default=640, help='推理大小 (像素)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='对象置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS的IoU阈值')
    parser.add_argument('--device', default='', help='cuda设备, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='按类别过滤')
    parser.add_argument('--names', type=str, default='data/coco.names', help='类别名称文件')
    return parser.parse_args()

def detect(model, source, output_dir, img_size, conf_thres, iou_thres, device, class_names, classes=None):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    model = model.to(device)
    model.eval()
    
    # 确定输入类型（图像、视频或目录）
    if os.path.isdir(source):
        # 目录
        files = [os.path.join(source, f) for f in os.listdir(source) 
                if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov'))]
    elif os.path.isfile(source):
        # 单个文件
        files = [source]
    else:
        raise ValueError(f"无法识别的源: {source}")
    
    # 处理每个文件
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"处理: {file_name}")
        
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 处理图像
            process_image(model, file_path, output_dir, img_size, conf_thres, iou_thres, device, class_names, classes)
        elif file_path.endswith(('.mp4', '.avi', '.mov')):
            # 处理视频
            process_video(model, file_path, output_dir, img_size, conf_thres, iou_thres, device, class_names, classes)

def process_image(model, img_path, output_dir, img_size, conf_thres, iou_thres, device, class_names, classes=None):
    # 加载图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 准备输入
    img_tensor, ratio, pad = prepare_input(img, img_size, device)
    
    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        # 处理输出
        outputs = [output.view(output.shape[0], -1, output.shape[-1]) for output in outputs]
        outputs = torch.cat(outputs, 1)
        # 应用NMS
        outputs = non_max_suppression(outputs, conf_thres, iou_thres, classes)
    
    # 处理检测结果
    if outputs[0] is not None:
        # 转换坐标到原始图像
        detections = outputs[0].cpu().numpy()
        detections[:, :4] = scale_coords(detections[:, :4], img.shape, ratio, pad)
        
        # 绘制结果
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        plot_boxes(img, detections, class_names, output_path)
        print(f"结果保存到: {output_path}")
    else:
        print("未检测到对象")

def process_video(model, video_path, output_dir, img_size, conf_thres, iou_thres, device, class_names, classes=None):
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建输出视频
    output_path = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理每一帧
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 准备输入
        img_tensor, ratio, pad = prepare_input(frame_rgb, img_size, device)
        
        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            # 处理输出
            outputs = [output.view(output.shape[0], -1, output.shape[-1]) for output in outputs]
            outputs = torch.cat(outputs, 1)
            # 应用NMS
            outputs = non_max_suppression(outputs, conf_thres, iou_thres, classes)
        
        # 处理检测结果
        if outputs[0] is not None:
            # 转换坐标到原始图像
            detections = outputs[0].cpu().numpy()
            detections[:, :4] = scale_coords(detections[:, :4], frame_rgb.shape, ratio, pad)
            
            # 在帧上绘制检测框
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                label = f"{class_names[int(cls)]}: {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 写入输出视频
        out.write(frame)
        
        # 更新进度
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"处理帧: {frame_count}/{total_frames}")
    
    # 释放资源
    cap.release()
    out.release()
    print(f"结果保存到: {output_path}")

def prepare_input(img, img_size, device):
    """准备模型输入"""
    h, w, _ = img.shape
    
    # 计算缩放比例
    ratio = min(img_size / h, img_size / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # 调整图像大小
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建新图像（填充黑色）
    new_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # 计算填充
    pad_h, pad_w = (img_size - new_h) // 2, (img_size - new_w) // 2
    
    # 将调整大小的图像放在中心
    new_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized_img
    
    # 转换为张量
    img_tensor = torch.from_numpy(new_img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    return img_tensor, ratio, (pad_w, pad_h)

def scale_coords(coords, img_shape, ratio, pad):
    """将坐标从模型输出缩放到原始图像"""
    pad_w, pad_h = pad
    
    # 移除填充
    coords[:, [0, 2]] -= pad_w
    coords[:, [1, 3]] -= pad_h
    
    # 缩放回原始大小
    coords[:, :4] /= ratio
    
    # 裁剪到图像边界
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, img_shape[1])
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, img_shape[0])
    
    return coords

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.device:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device.isdigit() else 'cpu')
    
    # 加载类别名称
    class_names = load_classes(args.names)
    num_classes = len(class_names)
    
    # 创建模型
    model = YOLOv11(num_classes=num_classes)
    
    # 加载权重
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f'加载权重: {args.weights}')
    
    # 运行检测
    detect(
        model, args.source, args.output, args.img_size,
        args.conf_thres, args.iou_thres, device, class_names, args.classes
    )

if __name__ == '__main__':
    main()