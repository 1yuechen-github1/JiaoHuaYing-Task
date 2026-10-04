import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import cv2
# from nnunetv2.inference.predict import nnUNetPredictor
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from sklearn.cluster import DBSCAN
from glob import glob
import re
# import torch

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 读取点云数据
def load_point_cloud_from_txt(txt_file):
    logging.info(f"加载点云文件: {txt_file}")
    points = []
    colors = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  
                values = list(map(float, line.split()))
                if len(values) >= 6:  
                    points.append(values[:3])  
                    colors.append(values[3:6])  
    points = np.array(points)
    colors = np.array(colors) / 255.0  
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    logging.info(f"加载点云成功: {txt_file}, 点数: {len(points)}")
    return point_cloud, points, colors

def load_point_cloud_from_pcd(pcd_file):
    logging.info(f"加载点云文件: {pcd_file}")

    point_cloud = o3d.io.read_point_cloud(pcd_file)
    if point_cloud.is_empty():
        raise ValueError(f"点云为空或读取失败: {pcd_file}")

    points = np.asarray(point_cloud.points)

    colors = np.asarray(point_cloud.colors)
    if len(colors) != len(points):
        colors = np.ones((len(points), 3), dtype=np.float64)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    logging.info(f"加载点云成功: {pcd_file}, 点数: {len(points)}")
    return point_cloud, points, colors

# 旋转点云
def rotate_point_cloud(point_cloud, axis='y', angle_deg=0):
    logging.info(f"旋转点云，绕 {axis} 轴旋转 {angle_deg} 度")
    angle_rad = np.radians(angle_deg)
    if axis == 'x':
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([angle_rad, 0, 0])
    elif axis == 'y':
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle_rad, 0])
    else:  # 'z'
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
    rotated = point_cloud.rotate(rot, center=(0, 0, 0))
    # logging.info(f"旋转完成，绕 {axis} 轴旋转 {angle_deg} 度")
    return rotated

# 批量加载点云文件
def load_all_point_clouds_from_folder(folder_path):
    logging.info(f"开始加载文件夹: {folder_path}")
    point_clouds = []
    original_points = []
    original_colors = []
    filenames = []
    for root,dirs,files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ply"):
                if 'lower' in file or 'Lower' in file or 'upper' in file or 'Upper' in file or 'Upp' in file or 'upp' in file or 'low' in file or 'Low' in file:
                    pcd_file = os.path.join(root, file)
                    point_cloud, points, colors = load_point_cloud_from_pcd(pcd_file)
                    point_clouds.append(point_cloud)
                    original_points.append(points)
                    original_colors.append(colors)
                    filenames.append(file)

    logging.info(f"共加载 {len(filenames)} 个点云文件")
    return point_clouds, original_points, original_colors, filenames


# def project_and_save_image(point_cloud, save_path, img_size=800):
#     # logging.info(f"开始生成精确投影图: {save_path}")
#     points = np.asarray(point_cloud.points)
#     colors = np.asarray(point_cloud.colors)
#     x_coords = points[:, 0]
#     y_coords = points[:, 1]
#     rgb_colors = (colors * 255).astype(np.uint8)
#     # 计算点云范围
#     x_min, x_max = np.min(x_coords), np.max(x_coords)
#     y_min, y_max = np.min(y_coords), np.max(y_coords)
#     range_x = x_max - x_min
#     range_y = y_max - y_min
#     if range_x > range_y:
#         scale = (img_size - 1) / range_x
#         # 在Y方向居中
#         y_offset = (img_size - range_y * scale) / 2
#         x_offset = 0
#     else:
#         scale = (img_size - 1) / range_y
#         # 在X方向居中
#         x_offset = (img_size - range_x * scale) / 2
#         y_offset = 0
#     pixel_x = ((x_coords - x_min) * scale + x_offset).astype(int)
#     pixel_y = ((y_coords - y_min) * scale + y_offset).astype(int)
#     pixel_y = img_size - 1 - pixel_y
#     image = np.zeros((img_size, img_size, 3), dtype=np.uint8) + 255  # 白色背景
#     valid_indices = (pixel_x >= 0) & (pixel_x < img_size) & (pixel_y >= 0) & (pixel_y < img_size)
#     pixel_x_valid = pixel_x[valid_indices]
#     pixel_y_valid = pixel_y[valid_indices]
#     colors_valid = rgb_colors[valid_indices]
    
#     # 设置点的大小（可以根据需要调整）
#     point_size = 2
#     for i in range(len(pixel_x_valid)):
#         x, y = pixel_x_valid[i], pixel_y_valid[i]
#         # 绘制一个小圆点
#         cv2.circle(image, (x, y), point_size, colors_valid[i].tolist(), -1)
#     # 保存图像
#     cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#     logging.info(f"精确投影图已保存: {save_path}")
    
#     # 返回映射参数
#     mapping_info = {
#         'x_min': x_min,
#         'y_min': y_min,
#         'x_max': x_max,
#         'y_max': y_max,
#         'scale': scale,
#         'x_offset': x_offset,
#         'y_offset': y_offset,
#         'img_size': img_size,
#         'valid_indices': valid_indices  # 有效点的索引
#     }
    
#     return save_path, mapping_info


# def crop_point_cloud_with_segmentation(original_points, original_colors, segmentation_mask, mapping_info, output_path, file_name_without_ext):
#     """
#     根据nnU-Net分割结果裁剪原始点云（使用精确映射）
    
#     Parameters:
#     original_points: 原始点云坐标 (N, 3)
#     original_colors: 原始点云颜色 (N, 3)
#     segmentation_mask: 分割掩码图像
#     mapping_info: 映射信息字典，包含精确映射参数
#     output_path: 输出点云文件路径
#     file_name_without_ext: 文件名（不含扩展名）
#     """
#     # 从映射信息中提取参数
#     x_min = mapping_info['x_min']
#     y_min = mapping_info['y_min']
#     x_max = mapping_info['x_max']
#     y_max = mapping_info['y_max']
#     scale = mapping_info['scale']
#     x_offset = mapping_info['x_offset']
#     y_offset = mapping_info['y_offset']
#     img_size = mapping_info['img_size']
#     valid_indices_projection = mapping_info['valid_indices']
    
#     # 调整分割掩码大小以匹配投影图像
#     if segmentation_mask.shape[:2] != (img_size, img_size):
#         segmentation_mask = cv2.resize(segmentation_mask, (img_size, img_size), 
#                                      interpolation=cv2.INTER_NEAREST)
#     x_coords = ((original_points[:, 0] - x_min) * scale + x_offset).astype(int)
#     y_coords = ((original_points[:, 1] - y_min) * scale + y_offset).astype(int)
#     y_coords = img_size - 1 - y_coords
#     valid_mask = (x_coords >= 0) & (x_coords < img_size) & (y_coords >= 0) & (y_coords < img_size)
#     if valid_indices_projection is not None:
#         valid_mask = valid_mask & valid_indices_projection
#     point_labels = np.zeros(len(original_points), dtype=np.uint8)
#     valid_indices = np.where(valid_mask)[0]
#     if len(valid_indices) > 0:
#         point_labels[valid_indices] = segmentation_mask[y_coords[valid_indices], x_coords[valid_indices]]
#     # 选择被分割为目标区域的点（假设目标区域的标签为1, 2, 3）
#     target_mask = np.isin(point_labels, [1, 2, 3])
#     cropped_points = original_points[target_mask]
#     if len(cropped_points) < 15000:
#         logging.warning(f"点云数量不足 {len(cropped_points)} < 15000，跳过处理")
#         return None, None  # 或者返回空结果
#     else:
#         cropped_colors = original_colors[target_mask]
#         cropped_labels = point_labels[target_mask]
    
#     unique_labels, counts = np.unique(point_labels, return_counts=True)
#     logging.info(f"标签统计: {dict(zip(unique_labels, counts))}")
#     logging.info(f"有效点数量: {np.sum(valid_mask)}/{len(original_points)}")
#     logging.info(f"目标点数量: {len(cropped_points)}")

#     if len(unique_labels) > 1:
#         for label in unique_labels:
#             if label == 0:
#                 continue  # 跳过背景或无效标签
#             label_mask = (cropped_labels == label)
#             points_in_label = cropped_points[label_mask]
#             points_in_color = cropped_colors[label_mask]
#             if len(points_in_label) == 0:
#                 continue  # 跳过空点云
#             output_filename = f"{file_name_without_ext}_{label}.txt"
#             output_filepath = os.path.join(output_path, output_filename)
#             save_cropped_point_cloud(points_in_label, points_in_color, output_filepath)
#     return cropped_points, cropped_colors



def project_and_save_image(
        point_cloud,
        save_path,
        img_size=800):

    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    rgb_colors = (colors * 255).astype(np.uint8)

    # ==========================================
    # 1. 计算范围
    # ==========================================
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)

    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    range_x = x_max - x_min
    range_y = y_max - y_min

    # ==========================================
    # 2. 计算缩放
    # ==========================================
    if range_x > range_y:

        scale = (img_size - 1) / range_x

        x_offset = 0.0
        y_offset = (
            img_size - range_y * scale
        ) / 2.0

    else:

        scale = (img_size - 1) / range_y

        x_offset = (
            img_size - range_x * scale
        ) / 2.0

        y_offset = 0.0

    # ==========================================
    # 3. 3D -> 2D
    # ==========================================
    pixel_x_float = (
        (x_coords - x_min)
        * scale
        + x_offset
    )

    pixel_y_float = (
        (y_coords - y_min)
        * scale
        + y_offset
    )

    # Y轴翻转
    pixel_y_float = (
        img_size - 1
        - pixel_y_float
    )

    # 使用 round 而不是直接 int
    pixel_x = np.round(
        pixel_x_float
    ).astype(np.int32)

    pixel_y = np.round(
        pixel_y_float
    ).astype(np.int32)

    # ==========================================
    # 4. 有效点
    # ==========================================
    valid_indices = (
        (pixel_x >= 0) &
        (pixel_x < img_size) &
        (pixel_y >= 0) &
        (pixel_y < img_size)
    )

    valid_point_indices = np.where(
        valid_indices
    )[0]

    # ==========================================
    # 5. 创建图片
    # ==========================================
    image = np.ones(
        (img_size, img_size, 3),
        dtype=np.uint8
    ) * 255

    # ==========================================
    # 6. 创建像素 -> 3D点索引映射
    # ==========================================

    # 注意：
    # 一个pixel可能对应多个3D点
    #
    # 所以不能使用：
    # pixel_index_map[y,x] = point_index
    #
    # 否则会覆盖。

    from collections import defaultdict

    pixel_to_points = defaultdict(list)

    for point_idx in valid_point_indices:

        x = pixel_x[point_idx]
        y = pixel_y[point_idx]

        pixel_to_points[
            (int(y), int(x))
        ].append(point_idx)

    # ==========================================
    # 7. 绘制点
    # ==========================================

    point_size = 2

    for point_idx in valid_point_indices:

        x = pixel_x[point_idx]
        y = pixel_y[point_idx]

        cv2.circle(
            image,
            (int(x), int(y)),
            point_size,
            rgb_colors[point_idx].tolist(),
            -1
        )

    # ==========================================
    # 8. 保存图片
    # ==========================================

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )
    )

    logging.info(
        f"精确投影图已保存: {save_path}"
    )

    # ==========================================
    # 9. 保存映射
    # ==========================================

    mapping_info = {

        "x_min": float(x_min),
        "x_max": float(x_max),

        "y_min": float(y_min),
        "y_max": float(y_max),

        "scale": float(scale),

        "x_offset": float(x_offset),
        "y_offset": float(y_offset),

        "img_size": int(img_size),

        "valid_indices":
            valid_indices,

        # 最重要
        "pixel_to_points":
            pixel_to_points,
    }

    return save_path, mapping_info


def crop_point_cloud_with_segmentation(
        original_points,
        original_colors,
        segmentation_mask,
        mapping_info,
        output_path,
        file_name_without_ext):

    """
    根据2D分割结果，精确映射回原始3D点云。

    一个2D pixel 可以对应多个3D点。
    """

    img_size = mapping_info["img_size"]

    pixel_to_points = mapping_info["pixel_to_points"]

    # ==========================================
    # 1. 保证 mask 尺寸正确
    # ==========================================

    if segmentation_mask.shape[:2] != (
            img_size,
            img_size):

        segmentation_mask = cv2.resize(
            segmentation_mask,
            (img_size, img_size),
            interpolation=cv2.INTER_NEAREST
        )

    # 如果是三通道
    if segmentation_mask.ndim == 3:
        segmentation_mask = segmentation_mask[:, :, 0]

    # ==========================================
    # 2. 找到 mask 中所有目标像素
    # ==========================================

    target_pixels = np.isin(
        segmentation_mask,
        [1, 2, 3]
    )

    target_y, target_x = np.where(
        target_pixels
    )

    # ==========================================
    # 3. 根据 pixel 找对应的3D点
    # ==========================================

    selected_indices = []

    point_labels = []

    for y, x in zip(
            target_y,
            target_x):

        key = (
            int(y),
            int(x)
        )

        if key not in pixel_to_points:
            continue

        point_indices = pixel_to_points[key]

        label = segmentation_mask[y, x]

        selected_indices.extend(
            point_indices
        )

        point_labels.extend(
            [label] * len(point_indices)
        )

    # ==========================================
    # 4. 没有点
    # ==========================================

    if len(selected_indices) == 0:

        logging.warning(
            "没有找到对应的3D点"
        )

        return None, None

    # ==========================================
    # 5. 转numpy
    # ==========================================

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64
    )

    point_labels = np.asarray(
        point_labels,
        dtype=np.uint8
    )

    # ==========================================
    # 6. 去除重复点
    # ==========================================

    unique_indices, unique_pos = np.unique(
        selected_indices,
        return_index=True
    )

    point_labels = point_labels[
        unique_pos
    ]

    selected_indices = unique_indices

    # ==========================================
    # 7. 获取3D点
    # ==========================================

    cropped_points = original_points[
        selected_indices
    ]

    cropped_colors = original_colors[
        selected_indices
    ]

    cropped_labels = point_labels

    # ==========================================
    # 8. 数量检查
    # ==========================================

    if len(cropped_points) < 15000:

        logging.warning(
            f"点云数量不足 "
            f"{len(cropped_points)} < 15000，跳过处理"
        )

        return None, None

    # ==========================================
    # 9. 标签统计
    # ==========================================

    unique_labels, counts = np.unique(
        cropped_labels,
        return_counts=True
    )

    logging.info(
        f"裁剪后标签统计: "
        f"{dict(zip(unique_labels, counts))}"
    )

    logging.info(
        f"目标点数量: "
        f"{len(cropped_points)}"
    )

    # ==========================================
    # 10. 分类别保存
    # ==========================================

    for label in unique_labels:

        if label == 0:
            continue

        label_mask = (
            cropped_labels == label
        )

        points_in_label = (
            cropped_points[label_mask]
        )

        colors_in_label = (
            cropped_colors[label_mask]
        )

        if len(points_in_label) == 0:
            continue

        output_filename = (
            f"{file_name_without_ext}_{label}.txt"
        )

        output_filepath = os.path.join(
            output_path,
            output_filename
        )

        save_cropped_point_cloud(
            points_in_label,
            colors_in_label,
            output_filepath
        )

    return (
        cropped_points,
        cropped_colors
    )


def process_and_split_point_clouds(directory_path, eps=0.1, min_samples=10, plot=False):
    """
    处理指定目录下的所有点云数据文件，自动进行DBSCAN聚类分割，并保存每个聚类结果。
    """
    
    origin_crop = os.path.join(directory_path, 'cropped')
    file_paths = glob(os.path.join(origin_crop, "*.txt"))
    results = []
    
    for file_path in file_paths:
        print(f"处理文件: {file_path}")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # 读取点云数据
        all_data = np.loadtxt(file_path)
        if len(all_data) == 0:
            print(f"文件 {file_path} 没有有效点云数据，跳过。")
            continue
        ranges = []
        axes = ['X', 'Y', 'Z']
        
        for i in range(3):  # 遍历x, y, z三个轴
            if all_data.shape[1] > i:  # 确保数据有这个维度
                coord_range = np.ptp(all_data[:, i])  # 极差
                ranges.append(coord_range)
            else:
                ranges.append(0)
        
        best_axis = np.argmax(ranges)
        # 使用 DBSCAN 聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(all_data[:, best_axis].reshape(-1, 1))
        
        # 获取不同聚类的点
        unique_labels = np.unique(labels)
        pc_list = [all_data[labels == label] for label in unique_labels if label != -1]  # 排除噪声点
        separation_info = {
            'axis': axes[best_axis],
            'unique_labels': unique_labels,
            'best_axis_index': best_axis,
            'num_clusters': len(pc_list)
        }
        print(f"自动检测到沿 {separation_info['axis']}-轴 分割，聚类数量: {separation_info['num_clusters']}")
        file_info = {
            'file': base_filename,
            'num_clusters': separation_info['num_clusters'],
            'cluster_info': []
        }
        
        for idx, pc in enumerate(pc_list):
            print(f"点云{idx+1}: {pc.shape[0]} 个点")
            # 保存点云到txt文件（包含xyz + rgb），文件名使用原始文件名加上聚类编号
            if pc.shape[0] < 10000:
                print(f"点云{idx+1} 点数少于10000，跳过保存。")
                continue
            output_path = os.path.join(directory_path, "classfy_cropped",f'{base_filename}_dbscan_{idx+1}.txt')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savetxt(output_path, pc, delimiter=' ', fmt='%.6f')
            print(f"点云{idx+1}已保存至: {output_path}")
            
            file_info['cluster_info'].append({
                'cluster_idx': idx + 1,
                'num_points': pc.shape[0],
                'output_path': output_path
            })
        
        results.append(file_info)
   
    return results


def process_and_split_point_dian_clouds(cropped_points, eps=0.1, min_samples=10, plot=False):
    all_data = cropped_points
    if len(all_data) == 0:
        print(f"文件 {file_path} 没有有效点云数据，跳过。")
        return
    # 计算每个坐标轴的范围
    ranges = []
    axes = ['X', 'Y', 'Z']
    
    for i in range(3):  # 遍历x, y, z三个轴
        if all_data.shape[1] > i:  # 确保数据有这个维度
            coord_range = np.ptp(all_data[:, i])  # 极差
            ranges.append(coord_range)
        else:
            ranges.append(0)
    best_axis = np.argmax(ranges)
    
    # 使用 DBSCAN 聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(all_data[:, best_axis].reshape(-1, 1))
    
    # 获取不同聚类的点
    unique_labels = np.unique(labels)
    pc_list = [all_data[labels == label] for label in unique_labels if label != -1]  # 排除噪声点
    
    # 分割信息
    separation_info = {
        'axis': axes[best_axis],
        'unique_labels': unique_labels,
        'best_axis_index': best_axis,
        'num_clusters': len(pc_list)
    }
    
    # 输出分割信息
    print(f"自动检测到沿 {separation_info['axis']}-轴 分割，聚类数量: {separation_info['num_clusters']}")
    file_info = {
        'file': base_filename,
        'num_clusters': separation_info['num_clusters'],
        'cluster_info': []
    }
    
    for idx, pc in enumerate(pc_list):
        print(f"点云{idx+1}: {pc.shape[0]} 个点")
        
        # 保存点云到txt文件（包含xyz + rgb），文件名使用原始文件名加上聚类编号
        if pc.shape[0] < 10000:
            print(f"点云{idx+1} 点数少于2000，跳过保存。")
            continue
        output_path = os.path.join(directory_path, "classfy_cropped",f'{base_filename}_{idx+1}.txt')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savetxt(output_path, pc, delimiter=' ', fmt='%.6f')
        print(f"点云{idx+1}已保存至: {output_path}")
        
        file_info['cluster_info'].append({
            'cluster_idx': idx + 1,
            'num_points': pc.shape[0],
            'output_path': output_path
        })
    
    results.append(file_info)
   
    return results


def save_cropped_point_cloud(points, colors, output_path):
    """保存裁剪后的点云到文件"""
    colors_255 = (colors * 255).astype(int)

    with open(output_path, 'w') as f:
        for i in range(len(points)):
            f.write(f"{points[i][0]} {points[i][1]} {points[i][2]} "
                   f"{colors_255[i][0]} {colors_255[i][1]} {colors_255[i][2]}\n")
    
    logging.info(f"裁剪点云已保存: {output_path}")

# 初始化nnU-Net预测器
def initialize_nnunet_predictor(model_folder, device='cuda'):
    """初始化nnU-Net预测器"""
    logging.info(f"初始化nnU-Net预测器，模型路径: {model_folder}")
    
    if device == 'cuda':
        torch.set_num_threads(1)
        device = torch.device('cuda')
    else:
        torch.set_num_threads(os.cpu_count())
        device = torch.device('cpu')
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=True,
        allow_tqdm=True
    )
    
    # 初始化模型
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=[0])
    
    return predictor


# def nnunet_predict_and_crop(predictor, image_paths, original_points_list, original_colors_list, img_info_list, output_folder, num_classes=3):

def nnunet_predict_and_crop(image_paths, original_points_list, original_colors_list, img_info_list, output_folder, num_classes=3):

    logging.info(f"开始nnU-Net预测，处理 {len(image_paths)} 张图像")
    
    cropped_results = []
    
    # # 为每个图像创建临时输入和输出文件夹
    # temp_input_folder = os.path.join(output_folder, 'temp_input')
    # temp_output_folder = os.path.join(output_folder, 'temp_output')
    # os.makedirs(temp_input_folder, exist_ok=True)
    # os.makedirs(temp_output_folder, exist_ok=True)
    
    # # 复制图像到临时输入文件夹（按照nnU-Net要求的格式）
    # for i, image_path in enumerate(image_paths):
    #     # 读取PNG图像并转换为nnU-Net期望的格式
    #     img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #     if img is None:
    #         logging.error(f"无法读取图像: {image_path}")
    #         continue
            
    #     # 如果图像有透明度通道，移除它
    #     if img.shape[-1] == 4:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    #     filename = os.path.basename(image_path)
    #     filename = os.path.splitext(filename)[0]
    #     # 保存为PNG格式（nnU-Net可以处理）
    #     output_image_path = os.path.join(temp_input_folder, filename+f'_0000.png')
    #     cv2.imwrite(output_image_path, img)
    
    # # 使用标准的predict_from_files方法进行预测
    # try:
    #     predictor.predict_from_files(
    #         temp_input_folder, 
    #         temp_output_folder, 
    #         save_probabilities=False,
    #         overwrite=True,
    #         num_processes_preprocessing=1,  # 减少进程数以避免问题
    #         num_processes_segmentation_export=1,
    #         num_parts=1, 
    #         part_id=0
    #     )
    # except Exception as e:
    #     logging.error(f"预测过程中出错: {e}")
    #     # 清理临时文件夹
    #     import shutil
    #     # shutil.rmtree(temp_input_folder, ignore_errors=True)
    #     # shutil.rmtree(temp_output_folder, ignore_errors=True)
    #     return cropped_results

    print(
        "列表长度：",
        len(image_paths),
        len(original_points_list),
        len(original_colors_list),
        len(img_info_list),
    )

    temp_output_folder_yisheng = r"Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\nnUnet2d\nnunet2d\kousao\labelsTr"
    # 处理预测结果
    for i, (image_path, original_points, original_colors, img_info) in enumerate(zip(image_paths, original_points_list, original_colors_list, img_info_list)):
        print('---------->')
        # 查找对应的预测结果文件
        
        filename = os.path.basename(image_path)

        # name = os.path.splitext(filename)[0]   
        # name = name.split('_')[0]              
        # filename = name + '.png'

        print("filename",filename)
        predicted_file = os.path.join(temp_output_folder_yisheng, filename)
        print("predicted_file",predicted_file)
        if not os.path.exists(predicted_file):
            logging.warning(f"未找到预测结果文件: {predicted_file}")
            continue
        
        # 读取分割结果
        segmentation_mask = cv2.imread(predicted_file, cv2.IMREAD_GRAYSCALE)
        if segmentation_mask is None:
            logging.error(f"无法读取分割结果: {predicted_file}")
            continue
        
        # 处理不连续的区域：找到所有连通区域，并为每个区域分配主要标签
        processed_mask = process_disconnected_regions(segmentation_mask, num_classes)
        
        # 加载原始 RGB 图像
        original_image = cv2.imread(image_path)
        
        # 创建一个全黑的输出图像
        output_image = np.zeros_like(original_image)

        # 提取分割区域并恢复原始颜色
        mask = processed_mask > 0  # 只选择大于 0 的区域（即被分割的目标区域）
        output_image[mask] = original_image[mask]
        image_name = os.path.basename(image_path)
        file_name_without_ext = os.path.splitext(image_name)[0]
        save_origin_folder = os.path.join(output_folder, 'cropped')
        unique_classes = np.unique(segmentation_mask)
        for class_id in unique_classes:
            if class_id == 0:  # 跳过背景
                continue
            class_mask = (segmentation_mask == class_id).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(class_mask)
            print(f"类别 {class_id} 发现 {num_labels-1} 个连通区域")
            os.makedirs(save_origin_folder, exist_ok=True)
            # 对每个连通区域单独处理
            for component_id in range(1, num_labels):
                # 创建当前组件的临时掩码
                component_mask = (labels == component_id).astype(np.uint8) * class_id
                # 调用点云裁剪函数
                cropped_points, cropped_colors = crop_point_cloud_with_segmentation(
                    original_points, 
                    original_colors, 
                    component_mask, 
                    img_info,
                    output_folder,
                    f"{file_name_without_ext}_{component_id}"
                )
            
        cropped_results.append((cropped_points, cropped_colors))
        
        # 保存带有原始颜色的分割结果图像
        # result_filename = os.path.join(output_folder, f'segmentation_{file_name_without_ext}.png')
        # cv2.imwrite(result_filename, output_image)
        # logging.info(f"分割结果已保存: {result_filename}")
    # import shutil
    # # 清理临时文件夹
    # shutil.rmtree(temp_input_folder, ignore_errors=True)
    # shutil.rmtree(temp_output_folder, ignore_errors=True)
    # return cropped_results


def process_disconnected_regions(mask, num_classes):
    """
    处理不连续的区域：找到所有连通区域，并为每个区域分配主要标签
    同时过滤掉面积小于20像素的区域
    
    参数:
    mask: 输入的分割掩码
    num_classes: 类别数量
    
    返回:
    处理后的掩码
    """
    # 创建输出掩码
    processed_mask = np.zeros_like(mask)
    for class_id in range(1, num_classes + 1):  # 从1开始，0通常是背景
        class_mask = (mask == class_id).astype(np.uint8)
        if np.any(class_mask):
            num_labels, labels = cv2.connectedComponents(class_mask)
            for label in range(1, num_labels):  # 跳过背景标签0
                region_mask = (labels == label)
                area = np.sum(region_mask)
                if area < 200:
                    continue  # 跳过这个小区域
                region_values = mask[region_mask]
                if len(region_values) > 0:
                    dominant_value = np.bincount(region_values).argmax()
                    processed_mask[region_mask] = dominant_value
    
    return processed_mask

# 判断上颌还是下颌
def classify_jaw_type(file_path):
    """
    该函数读取单个txt文件中的点云数据，根据z坐标的平均值来判断是上颌还是下颌，并返回分类结果。
    
    :param file_path: 单个txt文件路径
    :return: 分类结果（上颌或下颌）
    """
    def read_point_cloud_from_pcd(file_path):
        # data = np.loadtxt(file_path, usecols=(0, 1, 2))  # 只加载x, y, z列
        point_cloud = o3d.io.read_point_cloud(file_path)
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)  # 转换为open3d点云对象
        return pcd

    def classify_jaw_type(pcd):
        points = np.asarray(pcd.points)
        mean_z = np.mean(points[:, 2])
        if mean_z > 0:
            return "上颌"  # 上颌
        else:
            return "下颌"  # 下颌
    pcd = read_point_cloud_from_pcd(file_path)
    jaw_type = classify_jaw_type(pcd)
    return jaw_type

