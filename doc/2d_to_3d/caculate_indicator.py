import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff, cdist
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.metrics import precision_recall_curve, auc

warnings.filterwarnings('ignore')

def calculate_dice(gt, pred, class_id):
    """
    计算Dice系数
    """
    gt_mask = (gt == class_id)
    pred_mask = (pred == class_id)
    
    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask) + np.sum(pred_mask)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union

def calculate_iou(gt, pred, class_id):
    """
    计算IoU
    """
    gt_mask = (gt == class_id)
    pred_mask = (pred == class_id)
    
    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def calculate_hausdorff_distance(gt_points, pred_points, percentile=95):
    """
    计算Hausdorff距离
    """
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.inf
    
    # 计算双向Hausdorff距离
    hd1 = directed_hausdorff(gt_points, pred_points)[0]
    hd2 = directed_hausdorff(pred_points, gt_points)[0]
    hd = max(hd1, hd2)
    
    # 对于百分位HD，需要计算所有距离
    if percentile < 100:
        all_distances = []
        for point in gt_points:
            min_dist = np.min(cdist([point], pred_points))
            all_distances.append(min_dist)
        for point in pred_points:
            min_dist = np.min(cdist([point], gt_points))
            all_distances.append(min_dist)
        
        all_distances = np.sort(all_distances)
        percentile_index = int(len(all_distances) * percentile / 100)
        hd = all_distances[percentile_index]
    
    return hd

def calculate_asd(gt_points, pred_points):
    """
    计算平均表面距离 (Average Surface Distance)
    """
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.inf
    
    # GT到Pred的平均距离
    gt_to_pred_dists = []
    for point in gt_points:
        min_dist = np.min(cdist([point], pred_points))
        gt_to_pred_dists.append(min_dist)
    
    # Pred到GT的平均距离
    pred_to_gt_dists = []
    for point in pred_points:
        min_dist = np.min(cdist([point], gt_points))
        pred_to_gt_dists.append(min_dist)
    
    asd = (np.mean(gt_to_pred_dists) + np.mean(pred_to_gt_dists)) / 2
    return asd

def calculate_oa(gt, pred):
    """
    计算总体准确率 (Overall Accuracy)
    """
    return np.sum(gt == pred) / len(gt)

def calculate_ap(gt, pred, class_id):
    """计算平均精度(AP) - PR曲线下面积"""
    y_true = (gt == class_id).astype(int)
    y_score = (pred == class_id).astype(int)  # 或者使用预测概率
    
    if len(np.unique(y_true)) < 2:
        return 0.0
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def calculate_acc(gt, pred, class_id):
    """计算类别准确率"""
    class_correct = np.sum((gt == class_id) & (pred == class_id))
    class_total = np.sum(gt == class_id)
    return class_correct / class_total if class_total > 0 else 0.0
    
    return np.sum(pred[class_mask] == class_id) / np.sum(class_mask)

def evaluate_pointcloud_segmentation(file_path):
    """
    评估单个点云文件的分割结果
    """
    try:
        # 读取点云数据
        data = np.loadtxt(file_path)
        
        if data.shape[1] < 8:
            print(f"文件 {file_path} 列数不足8列")
            return None
        
        # 提取坐标、真实标签和预测标签
        coords = data[:, :3]  # 前三列是坐标
        gt_labels = data[:, 6].astype(int)  # 第7列是真实标签
        gt_labels[gt_labels > 0] = 1

        pred_labels = data[:, 7].astype(int)  # 第8列是预测标签
        
        # 获取所有类别
        all_classes = np.unique(np.concatenate([gt_labels, pred_labels]))
        all_classes = sorted(all_classes)
        
        results = {
            'file_name': Path(file_path).name,
            'total_points': len(gt_labels)
        }
        
        # 计算每个类别的指标
        class_ious = []
        class_dices = []
        class_aps = []
        class_accs = []
        
        for class_id in all_classes:
            # Dice
            dice = calculate_dice(gt_labels, pred_labels, class_id)
            class_dices.append(dice)
            
            # IoU
            iou = calculate_iou(gt_labels, pred_labels, class_id)
            class_ious.append(iou)
            
            # AP
            ap = calculate_ap(gt_labels, pred_labels, class_id)
            class_aps.append(ap)
            
            # ACC
            acc = calculate_acc(gt_labels, pred_labels, class_id)
            class_accs.append(acc)
            
            # 为每个类别保存指标
            results[f'class_{class_id}_dice'] = dice
            results[f'class_{class_id}_iou'] = iou
            results[f'class_{class_id}_ap'] = ap
            results[f'class_{class_id}_acc'] = acc
        
        # 计算平均指标
        results['mean_dice'] = np.mean(class_dices)
        results['mean_iou'] = np.mean(class_ious)
        results['mean_ap'] = np.mean(class_aps)
        results['mean_acc'] = np.mean(class_accs)
        
        # OA
        results['oa'] = calculate_oa(gt_labels, pred_labels)
        
        # 计算HD95和ASD（只对前景类别）
        foreground_classes = [c for c in all_classes if c != 0]  # 假设0是背景
        
        hd95_values = []
        asd_values = []
        
        for class_id in foreground_classes:
            gt_points = coords[gt_labels == class_id]
            pred_points = coords[pred_labels == class_id]
            
            if len(gt_points) > 0 and len(pred_points) > 0:
                # HD95
                hd95 = calculate_hausdorff_distance(gt_points, pred_points, percentile=95)
                hd95_values.append(hd95)
                
                # ASD
                asd_val = calculate_asd(gt_points, pred_points)
                asd_values.append(asd_val)
        
        results['hd95'] = np.mean(hd95_values) if hd95_values else np.inf
        results['asd'] = np.mean(asd_values) if asd_values else np.inf
        
        print(f"处理完成: {Path(file_path).name}")
        return results
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def batch_evaluate_pointclouds(pointcloud_dir, output_excel_path):
    """
    批量评估点云分割结果
    """
    pointcloud_files = list(Path(pointcloud_dir).glob("*.txt"))
    
    all_results = []
    
    for file_path in pointcloud_files:
        results = evaluate_pointcloud_segmentation(str(file_path))
        if results is not None:
            all_results.append(results)
    
    if not all_results:
        print("没有找到有效的评估结果")
        return
    
    # 转换为DataFrame
    df_results = pd.DataFrame(all_results)
    
    # 重新排列列的顺序，让重要指标在前面
    columns_order = ['file_name', 'total_points', 'oa', 'mean_iou', 'mean_dice', 'mean_ap', 'mean_acc', 'hd95', 'asd']
    
    # 添加其他列
    other_columns = [col for col in df_results.columns if col not in columns_order]
    final_columns = columns_order + other_columns
    
    df_results = df_results[final_columns]
    
    # 保存到Excel
    df_results.to_excel(output_excel_path, index=False)
    print(f"\n评估完成! 结果已保存到: {output_excel_path}")
    
    # 打印总体统计
    print("\n总体统计:")
    print(f"处理文件数量: {len(all_results)}")
    print(f"平均OA: {df_results['oa'].mean():.4f}")
    print(f"平均mIoU: {df_results['mean_iou'].mean():.4f}")
    print(f"平均Dice: {df_results['mean_dice'].mean():.4f}")
    print(f"平均HD95: {df_results['hd95'].mean():.4f}")
    print(f"平均ASD: {df_results['asd'].mean():.4f}")

# 使用方法
if __name__ == "__main__":
    # 配置路径
    pointcloud_directory = r"Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\shanghouya"  # 点云文件目录
    output_excel_path = r"Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\shanghouya\evaluation_results.xlsx"  # 输出Excel文件路径
    
    # 批量评估
    batch_evaluate_pointclouds(pointcloud_directory, output_excel_path)


    