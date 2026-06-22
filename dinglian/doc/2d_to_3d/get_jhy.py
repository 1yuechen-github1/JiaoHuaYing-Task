
import os
import re
import numpy as np
import open3d as o3d


def get_jhy(ply_path, save_dir):
    for root, _, files in os.walk(ply_path):
        for file in files:
            if file.endswith(".ply"):
                if not re.search(r"upper|lower|Upper|Lower", file):
                    save_path = os.path.join(save_dir, os.path.splitext(file)[0] + ".txt")
                    pcd = o3d.io.read_point_cloud(os.path.join(root, file))
                    points = np.asarray(pcd.points)
                    np.savetxt(save_path, points, fmt="%.3f")
                    print(f"Saved {file} to {save_path}")


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


def ensure_xyzrgb(points, name):
    points = np.asarray(points, dtype=float)
    points = np.atleast_2d(points)

    if points.shape[1] < 3:
        raise ValueError(f"{name} 至少需要3列")

    if points.shape[1] < 6:
        rgb = np.ones((points.shape[0], 3)) * 255
        points = np.hstack([points[:, :3], rgb])
    else:
        points = points[:, :6]

    return points


def match_files(path, prefix):
    matches = []
    for file in os.listdir(path):
        if file.startswith(prefix):
            matches.append(file)
    return sorted(matches)


def load_points_any(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        arr = np.loadtxt(path)
        arr = np.atleast_2d(arr)
        return arr

    elif ext == ".ply":
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

    else:
        raise ValueError(f"不支持的文件格式: {path}")


def load_xyzrgb(path):
    data = load_points_any(path)
    return ensure_xyzrgb(data, os.path.basename(path))


def is_valid_jhy_txt(path, min_positive_points=20000):
    data = np.loadtxt(path)
    data = np.atleast_2d(data)

    if data.shape[1] < 7:
        return False, 0

    scalar = data[:, -1].astype(int)
    positive_count = int(np.sum(scalar > 0))

    return positive_count > min_positive_points, positive_count


def select_jhy_file(path_jiaohua, prefix, min_positive_points=20000):
    candidates = match_files(path_jiaohua, prefix)

    if not candidates:
        return None

    for file in candidates:
        full_path = os.path.join(path_jiaohua, file)
        ext = os.path.splitext(file)[1].lower()

        try:
            if ext == ".txt":
                ok, positive_count = is_valid_jhy_txt(full_path, min_positive_points)
                if ok:
                    return file
                print(f"跳过 {file}: scalar>0 点数不足，当前为 {positive_count}")

            elif ext == ".ply":
                return file

        except Exception as e:
            print(f"跳过 {file}: {e}")
            continue

    return None


def find_close_points(source_points, target_points, radius=0.01):
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points[:, :3])
    kdtree = o3d.geometry.KDTreeFlann(target_pcd)

    mask = np.zeros(source_points.shape[0], dtype=bool)
    for idx, point in enumerate(source_points[:, :3]):
        count, _, _ = kdtree.search_radius_vector_3d(point, radius)
        mask[idx] = count > 0
    return mask


def retoke_pcd_match(path_jiaohua, path_queya, out_path):
    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(path_queya):
        print(f"处理缺牙区文件: {file}")

        queyaqu = load_xyzrgb(os.path.join(path_queya, file))

        prefix = file[0:4]
        jiaohua_file = select_jhy_file(path_jiaohua, prefix, min_positive_points=20000)

        if jiaohua_file is None:
            print(f"未找到可用角化龈文件: {prefix}")
            continue

        print(f"匹配角化龈: {jiaohua_file}")

        jiaohua_path = os.path.join(path_jiaohua, jiaohua_file)
        jiaohua = load_xyzrgb(jiaohua_path)

        is_maxilla = jiaohua_file[-6:-4]
        if is_maxilla.isdigit() and int(is_maxilla) < 30:
            jiaohua = rotate_point_cloud(jiaohua, "y", 180)
            jiaohua = rotate_point_cloud(jiaohua, "x", -35)
            jiaohua = rotate_point_cloud(jiaohua, "z", 180)
        else:
            jiaohua = rotate_point_cloud(jiaohua, "x", -20)

        jiaohua = ensure_xyzrgb(jiaohua, "角化龈")

        mask = find_close_points(queyaqu, jiaohua, radius=1)

        label = int(file.split(".")[0][-1])
        print("label:", label)

        labels = np.zeros(queyaqu.shape[0], dtype=int)
        labels[mask] = label

        output = np.column_stack((queyaqu[:, :6], labels))
        save_path = os.path.join(out_path, os.path.splitext(file)[0] + ".txt")

        np.savetxt(
            save_path,
            output,
            fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d"],
        )

        print(f"已保存: {save_path}")






path_jiaohua = r"Z:\1.CY-SPACE\JiaoHuaYing\fei\newdata\linshi\jhy"
path_queya = r"Z:\1.CY-SPACE\JiaoHuaYing\fei\newdata\linshi\quyaqu"
out_path = r"Z:\1.CY-SPACE\JiaoHuaYing\fei\newdata\linshi\output"

retoke_pcd_match(path_jiaohua, path_queya, out_path)


def get_pcd_label(path):
    for file in os.listdir(path):
        if file.endswith(".txt"):
            data = np.loadtxt(os.path.join(path, file))
            coords = data[:, :3]
            labels = data[:, 6].astype(int)
            labels1 = labels[labels > 0] 
            print(file, np.unique(labels), labels.shape, labels1.shape)
            # print()

get_pcd_label(r"Z:\1.CY-SPACE\JiaoHuaYing\fei\newdata\linshi\output")