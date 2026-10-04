import os
import logging
import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_point_cloud_from_ply(ply_file):
    logging.info(f"加载点云文件: {ply_file}")

    ply = PlyData.read(ply_file)
    if "vertex" not in ply:
        logging.warning(f"PLY中没有vertex: {ply_file}")
        return o3d.geometry.PointCloud()

    vertex_data = ply["vertex"].data
    names = vertex_data.dtype.names

    if not all(k in names for k in ("x", "y", "z")):
        logging.warning(f"PLY缺少xyz坐标: {ply_file}")
        return o3d.geometry.PointCloud()

    points = np.vstack([
        vertex_data["x"],
        vertex_data["y"],
        vertex_data["z"]
    ]).T.astype(np.float64)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    if all(k in names for k in ("red", "green", "blue")):
        colors = np.vstack([
            vertex_data["red"],
            vertex_data["green"],
            vertex_data["blue"]
        ]).T.astype(np.float64) / 255.0
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        logging.info(f"加载成功: 点数={len(points)}, 颜色数={len(colors)}")
    else:
        logging.warning(f"PLY没有RGB字段: {ply_file}")
        point_cloud.colors = o3d.utility.Vector3dVector(
            np.zeros((len(points), 3), dtype=np.float64)
        )

    return point_cloud


def save_point_cloud_to_ply(point_cloud, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ok = o3d.io.write_point_cloud(output_path, point_cloud)
    if ok:
        logging.info(f"点云已保存到: {output_path}")
    else:
        logging.error(f"点云保存失败: {output_path}")


def rotate_point_cloud(point_cloud, axis='y', angle_deg=0):
    logging.info(f"旋转点云，绕 {axis} 轴旋转 {angle_deg} 度")
    angle_rad = np.radians(angle_deg)

    if axis == 'x':
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([angle_rad, 0, 0])
    elif axis == 'y':
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle_rad, 0])
    else:
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])

    rotated = o3d.geometry.PointCloud(point_cloud)
    rotated.rotate(rot, center=(0, 0, 0))
    return rotated


def load_all_point_clouds_from_folder(folder_path):
    logging.info(f"开始递归加载文件夹: {folder_path}")
    point_clouds = []
    filepaths = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".ply"):
                ply_file = os.path.join(root, filename)
                point_cloud = load_point_cloud_from_ply(ply_file)
                if not point_cloud.is_empty():
                    point_clouds.append(point_cloud)
                    filepaths.append(ply_file)

    logging.info(f"共加载 {len(filepaths)} 个点云文件")
    return point_clouds, filepaths


def project_and_save_image(point_cloud, save_path, img_size=800, point_size=2):
    logging.info(f"开始生成投影图: {save_path}")

    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    if len(points) == 0:
        logging.warning(f"点云为空，跳过投影: {save_path}")
        return None

    if colors.size == 0:
        colors = np.zeros((len(points), 3), dtype=np.float32)

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    rgb_colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    range_x = x_max - x_min
    range_y = y_max - y_min

    if range_x == 0 and range_y == 0:
        logging.warning(f"点云范围为0，无法投影: {save_path}")
        return None

    if range_x > range_y:
        scale = (img_size - 1) / max(range_x, 1e-8)
        y_offset = (img_size - range_y * scale) / 2
        x_offset = 0
    else:
        scale = (img_size - 1) / max(range_y, 1e-8)
        x_offset = (img_size - range_x * scale) / 2
        y_offset = 0

    pixel_x = ((x_coords - x_min) * scale + x_offset).astype(int)
    pixel_y = ((y_coords - y_min) * scale + y_offset).astype(int)
    pixel_y = img_size - 1 - pixel_y

    image = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    valid_mask = (
        (pixel_x >= 0) & (pixel_x < img_size) &
        (pixel_y >= 0) & (pixel_y < img_size)
    )

    pixel_x_valid = pixel_x[valid_mask]
    pixel_y_valid = pixel_y[valid_mask]
    colors_valid = rgb_colors[valid_mask]

    for x, y, color in zip(pixel_x_valid, pixel_y_valid, colors_valid):
        cv2.circle(image, (x, y), point_size, color.tolist(), -1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logging.info(f"投影图已保存: {save_path}")
    return save_path


def get_jaw_type_from_path(file_path):
    path_lower = file_path.lower()
    if "upper" in path_lower:
        return "upper"
    if "lower" in path_lower:
        return "lower"
    return None


input_folder = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\Add_2D_screenshot\pcd'
output_folder = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\Add_2D_screenshot\output'
rotated_output_folder = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\Add_2D_screenshot\output'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(rotated_output_folder, exist_ok=True)

point_clouds, filepaths = load_all_point_clouds_from_folder(input_folder)

for point_cloud, filepath in zip(point_clouds, filepaths):
    filename = os.path.basename(filepath)
    file_name_without_ext = os.path.splitext(filename)[0]

    jaw_type = get_jaw_type_from_path(filepath)
    if jaw_type is None:
        logging.warning(f"路径中既不包含 upper 也不包含 lower，跳过: {filepath}")
        continue

    if jaw_type == "upper":
        rotated_point_cloud = rotate_point_cloud(point_cloud, axis='y', angle_deg=180)
        rotated_point_cloud = rotate_point_cloud(rotated_point_cloud, axis='x', angle_deg=-35)
        rotated_point_cloud = rotate_point_cloud(rotated_point_cloud, axis='z', angle_deg=180)
    else:
        # -20 x
        rotated_point_cloud = rotate_point_cloud(point_cloud, axis='x', angle_deg=-20)


    rotated_ply_path = os.path.join(rotated_output_folder, f"{file_name_without_ext}.ply")
    save_point_cloud_to_ply(rotated_point_cloud, rotated_ply_path)

    save_png_path = os.path.join(output_folder, f"{file_name_without_ext}.png")
    project_and_save_image(rotated_point_cloud, save_png_path)

    logging.info(f"处理完成: {filepath}")
