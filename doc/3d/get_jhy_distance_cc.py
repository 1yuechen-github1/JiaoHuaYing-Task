import os
import re

import numpy as np
import open3d as o3d

# 先用较宽的最近距离得到候选点，再按候选点之间的空间连通性保留最大块。
# 单位必须与点云坐标单位一致；建议先从现有的 0.19 开始调 DISTANCE_THRESHOLD。
DISTANCE_THRESHOLD = 0.30
CONNECT_RADIUS = 0.60
MIN_COMPONENT_POINTS = 100


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


def nearest_distance_mask(source_points, target_points, distance_threshold):
    """返回每个 source 点到 target 点云最近点的距离及距离初筛掩码。"""
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points[:, :3])

    kdtree = o3d.geometry.KDTreeFlann(target_pcd)

    distances = np.full(source_points.shape[0], np.inf, dtype=np.float64)

    for idx, point in enumerate(source_points[:, :3]):
        count, _, squared_distances = kdtree.search_knn_vector_3d(point, 1)
        if count > 0:
            distances[idx] = np.sqrt(squared_distances[0])

    return distances <= distance_threshold, distances


def keep_largest_connected_component(points, candidate_mask, connect_radius,
                                     min_component_points=1):
    """在初筛阳性点中保留最大空间连通块，消除背景散块。"""
    candidate_indices = np.flatnonzero(candidate_mask)
    result_mask = np.zeros(len(points), dtype=bool)

    if len(candidate_indices) == 0:
        return result_mask, 0, 0

    candidate_xyz = points[candidate_indices, :3]
    candidate_pcd = o3d.geometry.PointCloud()
    candidate_pcd.points = o3d.utility.Vector3dVector(candidate_xyz)
    kdtree = o3d.geometry.KDTreeFlann(candidate_pcd)

    visited = np.zeros(len(candidate_indices), dtype=bool)
    largest_component = np.empty(0, dtype=np.int64)
    component_count = 0

    for start in range(len(candidate_indices)):
        if visited[start]:
            continue

        component_count += 1
        visited[start] = True
        stack = [start]
        component = []

        while stack:
            current = stack.pop()
            component.append(current)
            _, neighbors, _ = kdtree.search_radius_vector_3d(
                candidate_xyz[current], connect_radius
            )
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        if len(component) > len(largest_component):
            largest_component = np.asarray(component, dtype=np.int64)

    if len(largest_component) >= min_component_points:
        result_mask[candidate_indices[largest_component]] = True

    return result_mask, component_count, len(largest_component)


def transform_jiaohua(jiaohua, jiaohua_file):
    is_maxilla = jiaohua_file[-6:-4]

    if is_maxilla.isdigit() and int(is_maxilla) < 30:
        jiaohua = rotate_point_cloud(jiaohua, "y", 180)
        jiaohua = rotate_point_cloud(jiaohua, "x", -35)
        jiaohua = rotate_point_cloud(jiaohua, "z", 180)
    else:
        jiaohua = rotate_point_cloud(jiaohua, "x", -20)

    return ensure_xyzrgb(jiaohua, "角化龈")


def try_one_jiaohua_file(path_jiaohua, jiaohua_file, queyaqu, label,
                         distance_threshold=DISTANCE_THRESHOLD,
                         connect_radius=CONNECT_RADIUS):
    jiaohua_path = os.path.join(path_jiaohua, jiaohua_file)

    jiaohua = load_xyzrgb(jiaohua_path)
    jiaohua = transform_jiaohua(jiaohua, jiaohua_file)

    candidate_mask, distances = nearest_distance_mask(
        queyaqu, jiaohua, distance_threshold=distance_threshold
    )
    mask, component_count, largest_component_size = keep_largest_connected_component(
        queyaqu,
        candidate_mask,
        connect_radius=connect_radius,
        min_component_points=MIN_COMPONENT_POINTS,
    )

    print(
        f"distance <= {distance_threshold}: {candidate_mask.sum()} points | "
        f"components: {component_count} | largest: {largest_component_size} | "
        f"kept: {mask.sum()}"
    )

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
                output, labels, positive_count = try_one_jiaohua_file(
                    path_jiaohua=path_jiaohua,
                    jiaohua_file=jiaohua_file,
                    queyaqu=queyaqu,
                    label=label,
                    distance_threshold=DISTANCE_THRESHOLD,
                    connect_radius=CONNECT_RADIUS,
                )
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



path_jiaohua = r"Z:\1.CY-SPACE\JiaoHuaYing\三维核对后\三维核对后"
path_queya = r"Z:\1.CY-SPACE\JiaoHuaYing\三维核对后\quyaqu_旧"
out_path = r"Z:\1.CY-SPACE\JiaoHuaYing\三维核对后\quyaqu"

retoke_pcd_match(path_jiaohua, path_queya, out_path)
