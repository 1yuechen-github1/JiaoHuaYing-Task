# import numpy as np
# import os
# from pathlib import Path

# def add_labels_to_pointcloud(pointcloud_path, label_path, output_path):
#     """
#     将分割标签添加到点云数据的末尾
    
#     参数:
#         pointcloud_path: 点云txt文件路径
#         label_path: 分割标签npy文件路径  
#         output_path: 输出文件路径
#     """
#     # 读取点云数据
#     pointcloud = np.loadtxt(pointcloud_path)
    
#     # 读取分割标签
#     labels = np.load(label_path)
    
#     # 检查数据维度是否匹配
#     if len(pointcloud) != len(labels):
#         print(f"警告: 点云数据点数量({len(pointcloud)})与标签数量({len(labels)})不匹配!")
#         # 取较小值以避免错误
#         min_length = min(len(pointcloud), len(labels))
#         pointcloud = pointcloud[:min_length]
#         labels = labels[:min_length]
    
#     # 将标签添加到点云数据末尾
#     # 确保labels是1D数组，如果不是则进行转换
#     if labels.ndim > 1:
#         labels = labels.flatten()
    
#     # 将标签转为整型（如果是浮点数的话）
#     labels = labels.astype(int)
    
#     # 添加标签列
#     pointcloud_with_labels = np.column_stack((pointcloud, labels))
    
#     # 保存到新文件
#     np.savetxt(output_path, pointcloud_with_labels, fmt='%.6f', delimiter=' ')
    
#     print(f"成功处理: {pointcloud_path}")
#     print(f"原始点云形状: {pointcloud.shape}")
#     print(f"添加标签后形状: {pointcloud_with_labels.shape}")
#     print(f"已保存到: {output_path}")

# def batch_process_pointclouds(pointcloud_dir, label_dir, output_dir):
#     """
#     批量处理点云文件
    
#     参数:
#         pointcloud_dir: 点云txt文件目录
#         label_dir: 分割标签npy文件目录
#         output_dir: 输出目录
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 获取所有点云文件
#     pointcloud_files = list(Path(pointcloud_dir).glob("*.txt"))
    
#     processed_count = 0
    
#     for pc_file in pointcloud_files:
#         # 提取文件ID (0085) - 使用stem属性获取不带扩展名的文件名，然后分割
#         file_id = pc_file.stem.split('_')[0]  # 这里修正：使用stem而不是重新赋值pc_file
        
#         # 构建对应的标签文件路径
#         label_file = Path(label_dir) / f"{file_id}_pred.npy"  # 假设标签文件是 0085.npy
        
#         if not label_file.exists():
#             print(f"警告: 未找到对应的标签文件 {label_file}")
#             continue
        
#         # 构建输出文件路径
#         output_file = Path(output_dir) / f"Area_10_{file_id}_pred.txt"
        
#         # 处理单个文件
#         add_labels_to_pointcloud(str(pc_file), str(label_file), str(output_file))
#         processed_count += 1
    
#     print(f"\n批量处理完成! 共处理 {processed_count} 个文件")

# # 使用方法示例
# if __name__ == "__main__":
#     # 方法2: 批量处理
#     pointcloud_directory = r"C:\yuechen\code\jiaohuaying\2.data\1024\test"    # 点云文件目录
#     label_directory = r"C:\Users\yuechen\Desktop\result"      # 标签文件目录
#     output_directory = r"C:\yuechen\code\jiaohuaying\2.data\1024\test\output"     # 输出目录
#     batch_process_pointclouds(pointcloud_directory, label_directory, output_directory)

import numpy as np
import os
from pathlib import Path

def add_labels_to_pointcloud(pointcloud_path, labels, start_idx, output_path):
    """
    将分割标签添加到点云数据的末尾
    
    参数:
        pointcloud_path: 点云txt文件路径
        labels: 完整标签数组
        start_idx: 当前点云在标签数组中的起始索引
        output_path: 输出文件路径
    
    返回:
        end_idx: 处理后的结束索引
    """
    # 读取点云数据
    pointcloud = np.loadtxt(pointcloud_path)
    n_points = len(pointcloud)
    
    # 获取对应的标签片段
    end_idx = start_idx + n_points
    
    # 检查是否超出标签范围
    if end_idx > len(labels):
        print(f"警告: 标签数量不足! 需要{end_idx}个标签，但只有{len(labels)}个")
        # 调整到可用范围
        available_points = len(labels) - start_idx
        if available_points <= 0:
            print(f"错误: 没有可用的标签，跳过文件 {pointcloud_path}")
            return start_idx
        
        pointcloud = pointcloud[:available_points]
        n_points = available_points
        end_idx = len(labels)
    
    point_labels = labels[start_idx:start_idx + n_points]
    
    # 确保标签是1D数组
    if point_labels.ndim > 1:
        point_labels = point_labels.flatten()
    
    # 将标签转为整型
    point_labels = point_labels.astype(int)
    
    # 添加标签列
    pointcloud_with_labels = np.column_stack((pointcloud, point_labels))
    
    # 保存到新文件
    np.savetxt(output_path, pointcloud_with_labels, fmt='%.6f', delimiter=' ')
    
    print(f"成功处理: {Path(pointcloud_path).name}")
    print(f"  点云点数: {n_points}")
    print(f"  标签索引: {start_idx} - {end_idx-1}")
    print(f"  输出文件: {output_path}")
    
    return end_idx

def batch_process_pointclouds(pointcloud_dir, label_dir, output_dir):
    """
    批量处理点云文件
    
    参数:
        pointcloud_dir: 点云txt文件目录
        label_dir: 分割标签npy文件目录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有子文件夹（数字命名的文件夹）
    subdirs = [d for d in Path(pointcloud_dir).iterdir() if d.is_dir() and d.name.isdigit()]
    
    processed_count = 0
    
    for subdir in subdirs:
        file_id = subdir.name
        
        # 构建对应的标签文件路径
        label_file = Path(label_dir) / f"{file_id}_pred.npy"
        
        if not label_file.exists():
            print(f"警告: 未找到对应的标签文件 {label_file}")
            continue
        
        # 读取标签文件
        labels = np.load(label_file)
        print(f"\n处理文件夹: {file_id}, 标签总数: {len(labels)}")
        
        # 获取该文件夹下的所有点云文件（按文件名排序以确保顺序）
        pointcloud_files = sorted(subdir.glob("*.txt"))
        
        if not pointcloud_files:
            print(f"  文件夹 {file_id} 中没有找到txt文件")
            continue
        
        # 创建对应的输出子文件夹
        output_subdir = Path(output_dir) / file_id
        # output_subdir.mkdir(exist_ok=True)
        

        # 按顺序处理点云文件
        current_idx = 0
        for pc_file in pointcloud_files:
            # 构建输出文件路径（保持原文件名）
            output_file = Path(output_dir) / f"{pc_file.stem}_labels.txt"
            
            # 处理单个文件，获取下一个起始索引
            current_idx = add_labels_to_pointcloud(
                str(pc_file), 
                labels, 
                current_idx, 
                str(output_file)
            )
            
            processed_count += 1
            
            # 如果已经用完所有标签，提前结束
            if current_idx >= len(labels):
                print(f"  标签已用完，跳过剩余文件")
                break
    
    print(f"\n批量处理完成! 共处理 {processed_count} 个文件")

# 使用方法示例
if __name__ == "__main__":
    pointcloud_directory = r"C:\yuechen\code\jiaohuaying\1.code\0105\data\DATA\label\test"    # 点云文件目录
    label_directory = r"C:\yuechen\code\jiaohuaying\1.code\0105\data\DATA\label\test\test\result"      # 标签文件目录
    output_directory = r"C:\yuechen\code\jiaohuaying\1.code\0105\data\DATA\label\test\test\vis"     # 输出目录
    
    batch_process_pointclouds(pointcloud_directory, label_directory, output_directory)