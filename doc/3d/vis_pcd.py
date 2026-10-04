import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import os

def vis(pcd_list, file):
    o3d.visualization.draw_geometries(
        pcd_list,
        window_name=f"Point Clouds: {file}",
        width=800,
        height=600
    )

def smooth_labels(points, labels, k=20, iterations=3):
    """用近邻多数投票平滑 0/1 标签。"""
    points = np.asarray(points)
    labels = np.asarray(labels).reshape(-1).copy()

    tree = cKDTree(points)
    _, neighbor_ids = tree.query(points, k=min(k + 1, len(points)))
    neighbor_ids = neighbor_ids[:, 1:]

    binary_labels = (labels >= 1).astype(np.uint8)

    for _ in range(iterations):
        neighbor_target_ratio = binary_labels[neighbor_ids].mean(axis=1)

        # 超过一半邻居是目标点，则该点设为目标；否则设为非目标
        binary_labels = (neighbor_target_ratio >= 0.5).astype(np.uint8)

    return binary_labels


# --------------------------------
# 提取边界线 原始
# --------------------------------

# def get_surface_boundary(points, labels, target_label=1, k=12):
#     """
#     在三维点云表面找 target_label 与其它标签的交界点。

#     points: (N, 3)
#     labels: (N,)
#     返回:
#         boundary_points: (M, 3)，位于红蓝交界边中点的三维点
#     """
#     points = np.asarray(points)
#     labels = np.asarray(labels)

#     target_mask = labels >= target_label
#     tree = cKDTree(points)

#     # 每个点找自身之外的 k 个最近邻
#     _, neighbor_ids = tree.query(points, k=k + 1)
#     neighbor_ids = neighbor_ids[:, 1:]

#     boundary_points = []

#     # 只从红色点出发，寻找紧邻的非红色点
#     for i in np.where(target_mask)[0]:
#         for j in neighbor_ids[i]:
#             if not target_mask[j]:
#                 # 两点中点位于红蓝交界附近，并且仍在三维表面区域
#                 boundary_points.append((points[i] + points[j]) / 2.0)

#     boundary_points = np.asarray(boundary_points)

#     # 去除重复、非常相近的边界点
#     boundary_pcd = o3d.geometry.PointCloud()
#     boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
#     boundary_pcd = boundary_pcd.voxel_down_sample(voxel_size=0.15)

#     return np.asarray(boundary_pcd.points)


# ---------------------------------------
# 提取边界线 相比较原始 + 可操作边界线厚度
# ---------------------------------------

# def get_surface_boundary(points, labels, target_label=1, k=12, thickness=0.001):
#     points = np.asarray(points)
#     labels = np.asarray(labels).reshape(-1)

#     target_mask = labels >= target_label
#     tree = cKDTree(points)

#     _, neighbor_ids = tree.query(points, k=k + 1)
#     neighbor_ids = neighbor_ids[:, 1:]

#     # 先记录红色区域中真正贴着非红色区域的边界点索引
#     boundary_ids = set()

#     for i in np.where(target_mask)[0]:
#         for j in neighbor_ids[i]:
#             if not target_mask[j]:
#                 boundary_ids.add(i)
#                 break

#     if not boundary_ids:
#         return np.empty((0, 3), dtype=float)

#     boundary_ids = np.array(sorted(boundary_ids))
#     boundary_seed_points = points[boundary_ids]

#     # 向红色区域内部扩展 thickness 距离，形成“加厚边界带”
#     near_boundary_ids = tree.query_ball_point(
#         boundary_seed_points,
#         r=thickness,
#     )

#     thick_boundary_ids = set()
#     for ids in near_boundary_ids:
#         for index in ids:
#             if target_mask[index]:
#                 thick_boundary_ids.add(index)

#     thick_boundary_points = points[sorted(thick_boundary_ids)]

#     # 可选：体素下采样；体素越大，带越稀疏
#     boundary_pcd = o3d.geometry.PointCloud()
#     boundary_pcd.points = o3d.utility.Vector3dVector(thick_boundary_points)
#     boundary_pcd = boundary_pcd.voxel_down_sample(voxel_size=0.05)

#     return np.asarray(boundary_pcd.points)


# ---------------------------------------
# 提取边界线 相比较原始 不用 voxel_down_sample 
# 提升边界线平滑度
# ---------------------------------------

def get_surface_boundary(points, labels, target_label=1, k=12):
    points = np.asarray(points)

    # 先平滑 AI 标签
    smooth_binary = smooth_labels(points, labels, k=20, iterations=4)
    target_mask = smooth_binary > 0

    tree = cKDTree(points)
    _, neighbor_ids = tree.query(points, k=min(k + 1, len(points)))
    neighbor_ids = neighbor_ids[:, 1:]

    boundary_ids = []

    for i in np.where(target_mask)[0]:
        if np.any(~target_mask[neighbor_ids[i]]):
            boundary_ids.append(i)

    return points[np.asarray(boundary_ids)]



def save_to_mesh(merged_data,filename, output_dir):
    # merged_data 前三列是 x y z
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_data[:, :3])

    # 若第 4~6 列是 0~255 的 RGB，则保留颜色
    pcd.colors = o3d.utility.Vector3dVector(merged_data[:, 3:6] / 255.0)

    # 估计法线：半径要按模型单位调整
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=1.0,
            max_nn=30,
        )
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    # 计算点间平均距离，自动给 Ball Pivoting 设置重建半径
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_distance = distances.mean()

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([
            avg_distance * 1.5,
            avg_distance * 2.5,
            avg_distance * 4.0,
        ]),
    )

    mesh.compute_vertex_normals()

    mesh_path = os.path.join(
        output_dir,
        f"boundary_merged_{os.path.splitext(filename)[0]}.ply",
    )

    o3d.io.write_triangle_mesh(
        mesh_path,
        mesh,
        write_ascii=False,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )

    print("Mesh 已保存:", mesh_path)



def make_boundary_lines(boundary_points, k=2):
    """
    将边界点按最近邻连接成线。
    k=2 通常适合一条连续边界线。
    """
    boundary_points = np.asarray(boundary_points)

    if len(boundary_points) < 2:
        return None

    tree = cKDTree(boundary_points)
    distances, indices = tree.query(
        boundary_points,
        k=min(k + 1, len(boundary_points)),
    )

    lines = set()

    for i in range(len(boundary_points)):
        # 0 是点自身，从第 1 个最近邻开始
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):
            # 太远的点不连，避免跨过牙齿或空洞
            if dist < 0.5:  # 需要按你的点云单位调整
                lines.add(tuple(sorted((i, int(j)))))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(boundary_points)
    line_set.lines = o3d.utility.Vector2iVector(list(lines))
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 0.0], (len(lines), 1))
    )

    return line_set

def visualize_boundary(txt_path, target_label=1):
    # TXT: x y z r g b ai_label
    filename = os.path.basename(txt_path)
    data = np.loadtxt(txt_path)

    points = data[:, :3]
    gt = data[:, 6:7].astype(int).flatten()
    ai = data[:, 7:8].astype(int).flatten() 
    labels = data[:, 7:8].astype(int).flatten()


    boundary_points = get_surface_boundary(
        points,
        labels,
        target_label=target_label,
        k=12,
    )

    gt_boundary_points = get_surface_boundary(
        points,
        gt,
        target_label=target_label,
        k=12,
    )
    ai_boundary_points = get_surface_boundary(
        points,
        ai,
        target_label=target_label,
        k=12,
    )
    

    print("边界点数:", len(boundary_points))
    print('ai_boundary_points:', len(ai_boundary_points))
    print('gt_boundary_points:', len(gt_boundary_points))

    # 原始点云
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    colors = np.tile([0.2, 0.3, 1.0], (len(points), 1))
    colors[labels == target_label] = [1.0, 0.0, 0.0]
    cloud.colors = o3d.utility.Vector3dVector(colors)

    # 边界点：绿色，并放大显示
    ai_boundary = o3d.geometry.PointCloud()
    ai_boundary.points = o3d.utility.Vector3dVector(ai_boundary_points)
    ai_boundary.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 0.0], (len(ai_boundary_points), 1))
    )


    gt_boundary = o3d.geometry.PointCloud()
    gt_boundary.points = o3d.utility.Vector3dVector(gt_boundary_points)
    gt_boundary.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.0, 0.0], (len(gt_boundary_points), 1))
    )

    # 边界和原始点云叠加保存
    # 原始数据只取 xyz、rgb、label
    # 你的 label 在第 8 列，所以索引为 7
    original_data = data[:, :8]
    new_data = original_data.copy()

    # 每个原始点到最近边界点的距离
    ai_boundary_tree = cKDTree(ai_boundary_points)
    gt_boundary_tree = cKDTree(gt_boundary_points)
    ai_distances, _ = ai_boundary_tree.query(points, k=1)
    gt_distances, _ = gt_boundary_tree.query(points, k=1)

    # 依据点云单位调整；先试 0.05 或 0.1
    ai_boundary_mask = ai_distances < 0.05
    gt_boundary_mask = gt_distances < 0.05
    red_col = [255, 0, 0]
    green_col = [0, 255, 0]
    new_data[ai_boundary_mask, 3:6] = green_col
    new_data[gt_boundary_mask, 3:6] = red_col
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(new_data[:, 3:6] / 255.0)

    # 可视化点云的ai和gt的边界线
    # vis([pcd], filename)

    fill_data = original_data.copy()
    ai_mask = fill_data[:, 7] > 0
    gt_mask = fill_data[:, 6] > 0
    fill_data[ai_mask, 3:6] = green_col
    fill_data[gt_mask, 3:6] = red_col
    
    
    fill_pcd = o3d.geometry.PointCloud()
    fill_pcd.points = o3d.utility.Vector3dVector(fill_data[:, :3])
    fill_pcd.colors = o3d.utility.Vector3dVector(
        fill_data[:, 3:6] / 255.0
    )

    # vis([fill_pcd], filename)


    os.makedirs(os.path.join(output_dir, 'frame'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'fill'), exist_ok=True)
    # 保存为点云， ai和gt叠加，仅仅保留边界线
    output_path = os.path.join(
        output_dir,
        f"frame\\frame_{os.path.splitext(filename)[0]}.txt",
    )
    np.savetxt(
        output_path,
        new_data,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d", "%d"],
        comments="",
    )

    # 保存为点云， ai和gt叠加
    output_path = os.path.join(
        output_dir,
        f"fill\\fill_{os.path.splitext(filename)[0]}.txt",
    )
    np.savetxt(
        output_path,
        fill_data,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d", "%d"],
        comments="",
    )



    # save_to_mesh(merged_data, filename, output_dir)

    



if __name__ == "__main__":
    input_dir = r"E:\CY\JHY\JHY_HumanVsMachineMatch\角化龈分割结果_csj\人机比赛-csj"
    output_dir = r"E:\CY\JHY\JHY_HumanVsMachineMatch\角化龈分割结果_csj\人机比赛-csj_vis"
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            visualize_boundary(
                os.path.join(input_dir, file),
                target_label=1,
            )