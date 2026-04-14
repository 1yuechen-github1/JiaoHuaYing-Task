import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import cv2

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_point_cloud_to_txt(point_cloud, output_path):
    """将点云保存为TXT文件（格式: x y z r g b）"""
    points = np.asarray(point_cloud.points)
    colors = (np.asarray(point_cloud.colors) * 255).astype(int)
    data = np.hstack([points, colors])
    np.savetxt(output_path, data, fmt='%.6f %.6f %.6f %d %d %d')
    logging.info(f"点云已保存到: {output_path}")

# 读取点云数据 (假设txt文件每行至少有 x, y, z, r, g, b, s，但可以包含更多列)
def load_point_cloud_from_txt(txt_file):
    logging.info(f"加载点云文件: {txt_file}")
    points = []
    colors = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 忽略空行
                values = list(map(float, line.split()))
                if len(values) >= 6:  # 至少需要 6 个数来表示 x, y, z, r, g, b
                    points.append(values[:3])  # 取 x, y, z
                    colors.append(values[3:6])  # 取 r, g, b
    points = np.array(points)
    colors = np.array(colors) / 255.0  # 将颜色值归一化到 [0, 1]
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    logging.info(f"加载点云成功: {txt_file}")
    return point_cloud

# 使用旋转函数对点云进行旋转
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
    logging.info(f"旋转完成，绕 {axis} 轴旋转 {angle_deg} 度")
    return rotated

# 批量加载指定路径下的所有点云文件
def load_all_point_clouds_from_folder(folder_path):
    logging.info(f"开始加载文件夹: {folder_path}")
    point_clouds = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # 只处理txt文件
            txt_file = os.path.join(folder_path, filename)
            point_cloud = load_point_cloud_from_txt(txt_file)
            point_clouds.append(point_cloud)
            filenames.append(filename)  # 保存文件名
    logging.info(f"共加载 {len(filenames)} 个点云文件")
    return point_clouds, filenames



def project_and_save_image(point_cloud, save_path, img_size=800, point_size=2):
    logging.info(f"开始生成精确投影图: {save_path}")
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    
    # 获取XY坐标和颜色
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    rgb_colors = (colors * 255).astype(np.uint8)
    
    # 计算点云范围
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # 计算缩放比例（保持纵横比）
    range_x = x_max - x_min
    range_y = y_max - y_min
    
    if range_x > range_y:
        scale = (img_size - 1) / range_x
        y_offset = (img_size - range_y * scale) / 2
        x_offset = 0
    else:
        scale = (img_size - 1) / range_y
        x_offset = (img_size - range_x * scale) / 2
        y_offset = 0
    
    # 映射到像素坐标
    pixel_x = ((x_coords - x_min) * scale + x_offset).astype(int)
    pixel_y = ((y_coords - y_min) * scale + y_offset).astype(int)
    
    # 翻转Y轴（图像坐标原点在左上角）
    pixel_y = img_size - 1 - pixel_y
    
    # 创建空白图像（白色背景）
    image = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
    
    # 筛选有效坐标
    valid_mask = (pixel_x >= 0) & (pixel_x < img_size) & (pixel_y >= 0) & (pixel_y < img_size)
    pixel_x_valid = pixel_x[valid_mask]
    pixel_y_valid = pixel_y[valid_mask]
    colors_valid = rgb_colors[valid_mask]
    
    # 使用向量化操作绘制点（比循环更快）
    for x, y, color in zip(pixel_x_valid, pixel_y_valid, colors_valid):
        cv2.circle(image, (x, y), point_size, color.tolist(), -1)
    
    # 保存图像（注意颜色空间转换）
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logging.info(f"精确投影图已保存: {save_path}, 尺寸: {img_size}x{img_size}像素")
    
    
    return save_path


# 文件夹路径
input_folder = r'data\1106\oral_scan'  # 替换为实际的文件夹路径
output_folder = r'data\1106\rotake_oral_scan'  # 替换为实际的输出文件夹路径
rotated_output_folder = r"data\1106\rotake_oral_scan"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)
point_clouds, filenames = load_all_point_clouds_from_folder(input_folder)

for i, (point_cloud, filename) in enumerate(zip(point_clouds, filenames)):

    points = np.asarray(point_cloud.points)
    mean_z = np.mean(points[:, 2])

    if mean_z > 0:    
        # # 上颌
        rotated_point_cloud = rotate_point_cloud(point_cloud, axis='y', angle_deg=180)
        rotated_point_cloud = rotate_point_cloud(rotated_point_cloud, axis='x', angle_deg=-35)
        rotated_point_cloud = rotate_point_cloud(rotated_point_cloud, axis='z', angle_deg=180)
    else:
        # 下颌
        rotated_point_cloud = rotate_point_cloud(point_cloud, axis='x', angle_deg=-20)

    # 保存旋转后的点云
    rotated_filename = f"{filename}"
    save_point_cloud_to_txt(rotated_point_cloud, os.path.join(rotated_output_folder, rotated_filename))
    file_name_without_ext = os.path.splitext(filename)[0]  # 去除扩展名
    save_path = os.path.join(output_folder, f'{file_name_without_ext}.png')
    project_and_save_image(rotated_point_cloud, save_path)
    logging.info(f"处理完成: {filename}")

