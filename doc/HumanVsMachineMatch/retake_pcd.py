

import os
import re
import numpy as np
import open3d as o3d



def rotate_point_cloud(point_cloud, axis="y", angle_deg=0):
    point_cloud = np.asarray(point_cloud, dtype=float)
    point_cloud = np.atleast_2d(point_cloud)
    point_cloud = point_cloud[:, :3]

    angle_rad = np.radians(angle_deg)

    if axis == "x":
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([angle_rad, 0, 0])
    elif axis == "y":
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle_rad, 0])
    else:
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])

    return point_cloud @ rot.T




def load_points_any(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        arr = np.loadtxt(path)
        arr = np.atleast_2d(arr)
        return arr

    if ext == ".ply":
        pcd = o3d.io.read_point_cloud(path)

        if pcd.is_empty():
            raise ValueError(f"PLY为空或无法读取: {path}")

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        if colors is None or len(colors) != len(points):
            colors = np.ones((len(points), 3), dtype=float)

        if len(colors) > 0 and colors.max() <= 1.0:
            colors = colors * 255.0

        return np.hstack([points, colors])

    raise ValueError(f"不支持的文件格式: {path}")



def ensure_xyzrgb(points, name):
    points = np.asarray(points, dtype=float)
    points = np.atleast_2d(points)

    if points.shape[1] < 3:
        raise ValueError(f"{name} 至少需要3列坐标")

    if points.shape[1] < 6:
        rgb = np.ones((points.shape[0], 3), dtype=float) * 255
        points = np.hstack([points[:, :3], rgb])
    else:
        points = points[:, :6]

    return points




def load_xyzrgb(path):
    data = load_points_any(path)
    return ensure_xyzrgb(data, os.path.basename(path))



def transform_jiaohua(jiaohua, jiaohua_file):
    is_maxilla = jiaohua_file[-6:-4]

    if is_maxilla.isdigit() and int(is_maxilla) < 30:
        jiaohua = rotate_point_cloud(jiaohua, "y", 180)
        jiaohua = rotate_point_cloud(jiaohua, "x", -35)
        jiaohua = rotate_point_cloud(jiaohua, "z", 180)
    else:
        jiaohua = rotate_point_cloud(jiaohua, "x", -20)

    return ensure_xyzrgb(jiaohua, "角化龈")




file_dir = r'E:\CY\JHY\JHY_HumanVsMachineMatch\角化龈分割结果_sy\测试集-诗语\linshi'
for file in os.listdir(file_dir):
    if file.endswith('.ply'):
        path =os.path.join(file_dir,file)
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"PLY为空或无法读取: {path}")
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        data = np.hstack([points, colors])
    else:   
        data = np.loadtxt(os.path.join(file_dir,file))
        coord = data[:,0:3]
        color = data[:,3:6]


    transform_data = transform_jiaohua(data, file)

    save_dir = r"E:\CY\JHY\JHY_HumanVsMachineMatch\角化龈分割结果_sy\rotake_pcd"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        os.path.splitext(file)[0] + "_transformed.txt",
    )

    np.savetxt(
        save_path,
        transform_data,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"],
    )

    print("已保存:", save_path)

