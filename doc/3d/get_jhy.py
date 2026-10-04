
import os
import re
import numpy as np
import open3d as o3d


import os
import re
import numpy as np
import open3d as o3d


def get_jhy(ply_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

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
        raise ValueError(f"{name} 至少需要3列坐标")
    if points.shape[1] < 6:
        rgb = np.ones((points.shape[0], 3), dtype=float) * 255
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


def load_xyzrgb(path):
    data = load_points_any(path)
    return ensure_xyzrgb(data, os.path.basename(path))


def find_close_points(source_points, target_points, radius=1):
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points[:, :3])

    kdtree = o3d.geometry.KDTreeFlann(target_pcd)

    mask = np.zeros(source_points.shape[0], dtype=bool)

    for idx, point in enumerate(source_points[:, :3]):
        count, _, _ = kdtree.search_radius_vector_3d(point, radius)
        mask[idx] = count > 0

    return mask


def transform_jiaohua(jiaohua, jiaohua_file):
    is_maxilla = jiaohua_file[-6:-4]

    if is_maxilla.isdigit() and int(is_maxilla) < 30:
        jiaohua = rotate_point_cloud(jiaohua, "y", 180)
        jiaohua = rotate_point_cloud(jiaohua, "x", -35)
        jiaohua = rotate_point_cloud(jiaohua, "z", 180)
    else:
        jiaohua = rotate_point_cloud(jiaohua, "x", -20)

    return ensure_xyzrgb(jiaohua, "角化龈")


def try_one_jiaohua_file(path_jiaohua, jiaohua_file, queyaqu, label, radius=1):
    jiaohua_path = os.path.join(path_jiaohua, jiaohua_file)

    jiaohua = load_xyzrgb(jiaohua_path)
    jiaohua = transform_jiaohua(jiaohua, jiaohua_file)

    mask = find_close_points(queyaqu, jiaohua, radius=radius)

    labels = np.zeros(queyaqu.shape[0], dtype=int)
    labels[mask] = label

    output = np.column_stack((queyaqu[:, :6], labels))

    return output, labels, int(np.sum(labels > 0))


def retoke_pcd_match(path_jiaohua, path_queya, out_path):
    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(path_queya):
        if not file.lower().endswith((".txt", ".ply")):
            continue

        print(f"处理缺牙区文件: {file}")

        queya_path = os.path.join(path_queya, file)
        queyaqu = load_xyzrgb(queya_path)

        prefix = file[0:4]
        path_jiaohua = os.path.join(path_jiaohua,prefix)
        candidates = match_files(path_jiaohua, prefix)

        if len(candidates) == 0:
            print(f"未找到可用角化龈文件: {prefix}")
            continue

        label = int(os.path.splitext(file)[0][-1])
        print("label:", label)

        final_output = None
        final_file = None

        for jiaohua_file in candidates:
            try:
                output, labels, positive_count = try_one_jiaohua_file(path_jiaohua=path_jiaohua,jiaohua_file=jiaohua_file,queyaqu=queyaqu,label=label,radius=0.19)
                if len(np.unique(labels)) < 2:
                    print(f"跳过 {jiaohua_file}: 输出全为0")
                    continue

                final_output = output
                final_file = jiaohua_file
                print(f"使用角化龈文件: {jiaohua_file}, 匹配点数: {positive_count}")
                
                if positive_count < 7000:
                    print(f"警告: {jiaohua_file} 匹配点数过少 ({positive_count})，可能不完整")
                    continue
                break
                
            except Exception as e:
                print(f"跳过 {jiaohua_file}: {e}")
                continue

        if final_output is None:
            print(f"警告: {file} 的所有角化龈文件输出都是0，未保存")
            continue

        save_path = os.path.join(out_path, os.path.splitext(file)[0] + ".txt")

        np.savetxt(save_path,final_output,fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d"],)
        print(f"已保存: {save_path}")
        print(f"匹配来源: {final_file}")




path_jiaohua = r"Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\1.AllData"
path_queya = r"Z:\1.CY-SPACE\JiaoHuaYing\1.AllData-PointCloud-QueYaQu"
out_path = r"Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\quyaqu"




retoke_pcd_match(path_jiaohua, path_queya, out_path)

