import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def xywh2xyxy(x):
    """将边界框从[x, y, w, h]转换为[x1, y1, x2, y2]格式"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # 左上x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # 左上y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # 右下x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # 右下y
    return y

def xyxy2xywh(x):
    """将边界框从[x1, y1, x2, y2]转换为[x, y, w, h]格式"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # 中心x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # 中心y
    y[:, 2] = x[:, 2] - x[:, 0]  # 宽度
    y[:, 3] = x[:, 3] - x[:, 1]  # 高度
    return y

def box_iou(box1, box2):
    """计算两组边界框之间的IoU"""
    # 转换为xyxy格式
    b1 = xywh2xyxy(box1)
    b2 = xywh2xyxy(box2)
    
    # 交集区域
    inter_rect_x1 = torch.max(b1[:, 0].unsqueeze(1), b2[:, 0].unsqueeze(0))
    inter_rect_y1 = torch.max(b1[:, 1].unsqueeze(1), b2[:, 1].unsqueeze(0))
    inter_rect_x2 = torch.min(b1[:, 2].unsqueeze(1), b2[:, 2].unsqueeze(0))
    inter_rect_y2 = torch.min(b1[:, 3].unsqueeze(1), b2[:, 3].unsqueeze(0))
    
    # 交集面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    
    # 各自面积
    b1_area = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    b2_area = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    
    # 并集面积
    union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area
    
    # IoU
    iou = inter_area / union_area
    
    return iou

def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.45, classes=None):
    """非极大值抑制"""
    # 阈值筛选
    mask = prediction[..., 4] > conf_thres  # 置信度筛选
    output = [None] * len(prediction)
    
    for i, pred in enumerate(prediction):
        pred = pred[mask[i]]
        if not pred.size(0):
            continue
        
        # 计算类别得分 = 置信度 * 类别概率
        class_conf, class_pred = torch.max(pred[:, 5:], 1, keepdim=True)
        pred = torch.cat((pred[:, :5], class_conf, class_pred.float()), 1)
        
        # 按类别筛选
        if classes is not None:
            pred = pred[(pred[:, 6:7] == torch.tensor(classes, device=pred.device)).any(1)]
        
        # 如果没有检测结果，跳过
        if not pred.size(0):
            continue
        
        # 按置信度排序
        pred = pred[(-pred[:, 4]).argsort()]
        
        # 应用NMS
        det_max = []
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # 按类别选择
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # 如果只有一个检测结果，直接添加
                continue
            elif n > 100:  # 限制最大检测数量
                dc = dc[:100]
            
            while len(dc):
                det_max.append(dc[:1])  # 保留置信度最高的
                if len(dc) == 1:
                    break
                iou = box_iou(dc[0, :4].unsqueeze(0), dc[1:, :4])  # IoU与其他框
                dc = dc[1:][iou < iou_thres]  # 移除IoU > threshold的框
        
        if len(det_max):
            det_max = torch.cat(det_max)  # 合并
            output[i] = det_max
    
    return output

def plot_boxes(img, boxes, class_names, output_path=None):
    """绘制检测框和类别"""
    # 创建图形
    plt.figure(figsize=(12, 9))
    plt.imshow(img)
    ax = plt.gca()
    
    # 绘制每个边界框
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        width = x2 - x1
        height = y2 - y1
        
        # 创建矩形
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # 添加类别标签和置信度
        class_name = class_names[int(cls)] if class_names else f'Class {int(cls)}'
        plt.text(x1, y1, f'{class_name}: {conf:.2f}', color='white', fontsize=10,
                 bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')  # 关闭坐标轴
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()
    else:
        plt.show()

def load_classes(path):
    """加载类别名称"""
    with open(path, 'r') as f:
        names = f.read().strip().split('\n')
    return names