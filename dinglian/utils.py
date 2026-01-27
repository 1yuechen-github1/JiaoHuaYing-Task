import os
import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import Delaunay, cKDTree
from collections import Counter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# from code.dinglian.doc.surface_rebuild import mesh


def vis(pcd_list, file):
    o3d.visualization.draw_geometries(
        pcd_list,
        window_name=f"Point Clouds: {file}",
        width=800,
        height=600
    )

    

def filt_rpoin_hsv(pcd, hue_range1=(0, 25), hue_range2=(350, 360), 
                              saturation_threshold=0.3, value_threshold=0.3):
    """
    使用HSV颜色空间过滤，保留非红色点
    """
    colors = np.asarray(pcd.colors)
    hsv_colors = np.zeros_like(colors)
    for i, (r, g, b) in enumerate(colors):
        r, g, b = float(r), float(g), float(b)
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin
        if delta == 0:
            h = 0
        elif cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * ((b - r) / delta + 2)
        else:
            h = 60 * ((r - g) / delta + 4)
        if h < 0:
            h += 360
        s = 0 if cmax == 0 else delta / cmax
        v = cmax
        hsv_colors[i] = [h/360, s, v]
    
    # 提取HSV分量
    h_values = hsv_colors[:, 0] * 360
    s_values = hsv_colors[:, 1]
    v_values = hsv_colors[:, 2]
    is_red = (((h_values >= hue_range1[0]) & (h_values <= hue_range1[1])) | \
              ((h_values >= hue_range2[0]) & (h_values <= hue_range2[1]))) & \
             (s_values > saturation_threshold) & \
             (v_values > value_threshold)
    is_not_red = ~is_red
    points = np.asarray(pcd.points)
    not_red_points = points[is_not_red]
    not_red_colors = colors[is_not_red]
    not_red_pcd = o3d.geometry.PointCloud()
    not_red_pcd.points = o3d.utility.Vector3dVector(not_red_points)
    not_red_pcd.colors = o3d.utility.Vector3dVector(not_red_colors)

    points = np.asarray(pcd.points)
    red_points = points[is_red]
    red_colors = colors[is_red]
    red_pcd = o3d.geometry.PointCloud()
    red_pcd.points = o3d.utility.Vector3dVector(red_points)
    red_pcd.colors = o3d.utility.Vector3dVector(red_colors)
    return not_red_pcd, red_pcd


def use_dbscan(pcd):
    with o3d.utility.VerbosityContextManager(
         o3d.utility.VerbosityLevel.Debug) as cm:
     labels = np.array(
         pcd.cluster_dbscan(eps=1, min_points=10, print_progress=True))
    label_counts = Counter(labels[labels >= 0])    
    top_two_labels = [label for label, _ in label_counts.most_common(2)]
    mask = np.isin(labels, top_two_labels)
    pcd = pcd.select_by_index(np.where(mask)[0])
    remaining_labels = labels[mask]
    # 映射到新的标签
    label_mapping = {top_two_labels[0]: 0, top_two_labels[1]: 1}
    new_labels = np.array([label_mapping[label] for label in remaining_labels])
    colors = plt.get_cmap("tab20")(new_labels / 1)  # 只有0和1
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd,new_labels

def get_alx(center_x_list,pcd):
    # y = kx + b
    points = pcd.points 
    poin_list = []
    poin1 = center_x_list[0]
    poin2 = center_x_list[1]
    x1, y1 = poin1[0], poin1[1]
    x2, y2 = poin2[0], poin2[1]
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    for point in points:
        y = k * point[0] + b 
        if(y - point[1]< 0.01):
            poin_list.append(point)
    print('poin_list', len(poin_list))
    poin_array = np.array(poin_list)
    if k == float('inf'):
        sorted_indices = np.argsort(poin_array[:, 1])
    else:
        sorted_indices = np.argsort(poin_array[:, 0])
    sorted_points = poin_array[sorted_indices]
    point1 = sorted_points[0]
    point2 = sorted_points[-1]
    max_distance = np.linalg.norm(point1 - point2)
    dist_list = []
    dist_list.append(point1)
    dist_list.append(point2)
    return dist_list


def get_poin_list(poin_list, w_color = [0, 0, 1]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(poin_list)
    colors = np.array(w_color * len(poin_list))  
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_coordinate_frame(center, x_axis, y_axis, z_axis, scale=1.0):
    """
    创建自定义坐标系
    
    参数:
        center: 坐标系原点
        x_axis, y_axis, z_axis: 三个轴的方向向量
        scale: 坐标轴长度
    
    返回:
        coordinate_frame: 包含三个轴的Open3D几何体列表
    """
    center = np.array(center)
    x_axis = np.array(x_axis) * scale
    y_axis = np.array(y_axis) * scale
    z_axis = np.array(z_axis) * scale
    
    # 创建坐标轴线段
    axes = []
    
    # X轴 (黑色) Y轴 (绿色) Z轴 (蓝色)
    x_line = o3d.geometry.LineSet()
    x_line.points = o3d.utility.Vector3dVector([center, center + x_axis])
    x_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    x_line.colors = o3d.utility.Vector3dVector([[0, 0, 0]])  # 黑色
    
    #
    y_line = o3d.geometry.LineSet()
    y_line.points = o3d.utility.Vector3dVector([center, center + y_axis])
    y_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    y_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # 绿色
    
    #
    z_line = o3d.geometry.LineSet()
    z_line.points = o3d.utility.Vector3dVector([center, center + z_axis])
    z_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    z_line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # 蓝色
    axes.extend([x_line, y_line, z_line])
    return x_line, y_line, z_line



# X轴 (黑色) Y轴 (绿色) Z轴 (蓝色)
def get_jhy_w(jhy_points, cent_list, step_mm=1.0, vis_list_h =[],axiox_list = []):
    """
    只在中心及上下 1mm 位置做 3 个切片
    """
    sample_axis = cent_list[0]          # 采样方向（单位向量）
    vir_axis = cent_list[1]
    center = np.mean(jhy_points, axis=0)
    # 所有点在采样轴上的投影
    projections = np.dot(jhy_points, sample_axis)
    center_proj = np.dot(center, sample_axis)
    # 三个切片位置（mm）
    slice_positions = [
        center_proj,
        center_proj - step_mm,
        center_proj + step_mm,
        center_proj - step_mm * 2,
        center_proj + step_mm * 2,
        center_proj - step_mm * 3,
        center_proj + step_mm * 3
    ]
    tolerance = step_mm / 50.0  # 切片厚度
    pcd_list = []
    dist_list = []
    for pos in slice_positions:
        mask = np.abs(projections - pos) <= tolerance
        if np.any(mask):
            slice_points = jhy_points[mask]
            dist = get_len(slice_points,cent_list[1])
            dist_list.append(dist)
            pcd = get_poin_list(slice_points, [[0, 0, 1]])
            pcd_list.append(pcd)
        else:
            dist_list.append(0)
    return pcd_list, dist_list


def get_jhy_h(jhy_points, cent_list, step_mm=1.0,vis_list_h = [],axiox_list = []):
    """
    只在中心及上下 1mm 位置做 3 个切片
    """
    sample_axis = cent_list[1]          # 采样方向（单位向量）
    vir_axis = cent_list[0]
    center = np.mean(jhy_points, axis=0)
    # 所有点在采样轴上的投影
    projections = np.dot(jhy_points, sample_axis)
    center_proj = np.dot(center, sample_axis)
    slice_positions = [
        center_proj,
        center_proj - step_mm,
        center_proj + step_mm,
        center_proj - step_mm * 2,
        center_proj + step_mm * 2,
        center_proj - step_mm * 3,
        center_proj + step_mm * 3
    ]
    tolerance = step_mm / 50.0  # 切片厚度
    pcd_list = []
    dist_list = []
    for pos in slice_positions:
        mask = np.abs(projections - pos) <= tolerance
        if np.any(mask):
            slice_points = jhy_points[mask]
            dist = get_len(slice_points,cent_list[0])
            dist_list.append(dist)
            pcd = get_poin_list(slice_points, [[0, 0, 1]])
            pcd_list.append(pcd)
        else:
            dist_list.append(0)
    return pcd_list, dist_list

def get_len(points, axis):
    pts_centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(pts_centered, full_matrices=False)
    main_dir = vh[0]
    proj = np.dot(points, main_dir)
    sorted_points = points[np.argsort(proj)]
    diffs = np.diff(sorted_points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    med = np.median(dists)
    valid = dists[dists < med * 3]  # 去掉跳点
    print(np.sum(valid))
    return np.sum(valid)




# def get_len(points,axis, vis_list_h):
#     num_points = 1000
#     pts = np.asarray(points)
#     n = len(pts)
#     if n < 2:
#         return np.array([0.0]), pts, 0.0
#
#     # -------------------------
#     # 1️⃣ B样条拟合曲线
#     # -------------------------
#     tck, u = splprep(pts.T, s=0)  # s=0 保证拟合通过所有点
#     u_dense = np.linspace(0, 1, num_points)
#     dense_pts = np.array(splev(u_dense, tck)).T  # 拟合后的点
#     dense_pts = np.asarray(dense_pts)
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     vis_list_h.append(pcd)
#     vis(vis_list_h, " 三轴坐标系")
#     # vis_list_h.remove(pcd)
#     # -------------------------
#     # 2️⃣ 累计弧长（展平）
#     # -------------------------
#     diffs = np.diff(dense_pts, axis=0)
#     seg_len = np.linalg.norm(diffs, axis=1)
#     s = np.concatenate([[0], np.cumsum(seg_len)])
#     total_length = s[-1]
#     return total_length

# X轴 (黑色) Y轴 (绿色) Z轴 (蓝色)

# def get_len(points,axis, vis_list_h, axiox_list = []):
    # R = np.stack([axiox_list[0], axiox_list[1], axiox_list[2]], axis=1)
    # points = points @ R
    # print('np.allclose(R.T @ R, np.eye(3))',np.allclose(R.T @ R, np.eye(3)))

    points = np.asarray(points, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    order = np.argsort(points @ axis)
    ordered_pts = points[order]
    ordered_pts = points
    diffs = np.diff(ordered_pts, axis=0)
    length = np.sum(np.linalg.norm(diffs, axis=1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    vis_list_h.append(pcd)
    print("length", length)
    seg_lens = np.linalg.norm(diffs, axis=1)
    print("max segment:", seg_lens.max())
    print("mean segment:", seg_lens.mean())
    print("len segment:", len(diffs))
    # print("diffs segment:", diffs)
    vis(vis_list_h, " 三轴坐标系")
    return length


# def get_len(points, axis):
#     axis = axis / np.linalg.norm(axis)
#     order = np.argsort(points @ axis)
#     ordered_points = points[order]
#     diffs = np.diff(ordered_points, axis=0)
#     print(np.sum(np.linalg.norm(diffs, axis=1)))
#     return np.sum(np.linalg.norm(diffs, axis=1))

# def get_len(points,axis):
#     radius = len(points)
#     print('radius:',radius)
#     points = np.asarray(points)
#     # pcd = o3d.geometry.PointCloud()
#     # pcd.points = o3d.utility.Vector3dVector(points)
#     # vis([pcd],'file')
#     n = len(points)
#     G = nx.Graph()
#     kdt = cKDTree(points)
#     # radius 搜索邻居
#     for i, p in enumerate(points):
#         idxs = kdt.query_ball_point(p, r=radius)
#         for j in idxs:
#             if i != j:
#                 dist = np.linalg.norm(points[i] - points[j])
#                 G.add_edge(i, j, weight=dist)
#     # 检查是否连通
#     if not nx.has_path(G, 0, n - 1):
#         raise ValueError("图不连通，需增大 radius 拟合")
#     length = nx.shortest_path_length(G, 0, n - 1, weight='weight')
#     print("length", length)
#     return length

def save_to_txt(pcd2, pcd_list_h, output, file, scalar, status):
    points_array = np.asarray(pcd2.points)
    colors_array = np.asarray(pcd2.colors)
    colors_array = (colors_array * 255).astype(np.uint8)
    all_points = []
    all_colors = []
    has_colors = all(p.has_colors() for p in pcd_list_h if p is not None)
    for pcd in pcd_list_h:
        points = np.asarray(pcd.points)
        all_points.append(points)
        if has_colors and pcd.has_colors():
            colors = np.asarray(pcd.colors)
            colors = (colors * 255).astype(np.uint8)
            all_colors.append(colors)
    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = None
    if all_colors and len(all_colors) == len(all_points):
        combined_colors = np.concatenate(all_colors, axis=0)

    final_points = np.concatenate([combined_points, points_array],axis=0)
    final_colors = np.concatenate([combined_colors, colors_array],axis=0)
    num_combined = combined_points.shape[0]
    combined_scalar = np.zeros((num_combined, 1))
    scalar = scalar.reshape(-1, 1)

    final_scalar = np.concatenate([combined_scalar, scalar],axis=0)

    save_array = np.hstack([final_points, final_colors, final_scalar])
    os.makedirs(os.path.join(output,status,'txt'), exist_ok=True)
    np.savetxt(f"{output}\\{status}\\txt\\{file}",save_array,fmt="%.6f %.6f %.6f %.6f %.6f %.6f %.6f")


def interpolate_points_linear(points, num_samples=100):
    """
    线性插值增加点密度
    """
    # points = np.array(points)
    # n = len(points)
    #
    # if n < 2:
    #     return points
    # # 计算累积距离作为参数
    # cumulative_dist = np.zeros(n)
    # for i in range(1, n):
    #     cumulative_dist[i] = cumulative_dist[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    # # 归一化到 [0, 1]
    # t = cumulative_dist / cumulative_dist[-1]
    # # 对每个维度分别插值
    # interpolated_points = []
    # for dim in range(3):
    #     f = interp1d(t, points[:, dim], kind='linear')
    #     t_new = np.linspace(0, 1, num_interpolated)
    #     interpolated_dim = f(t_new)
    #     interpolated_points.append(interpolated_dim)
    points = np.asarray(points)
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)

    s = np.concatenate([[0], np.cumsum(seg_lens)])
    s_new = np.linspace(0, s[-1], num_samples)

    new_pts = []
    j = 0
    for si in s_new:
        while j < len(seg_lens) - 1 and si > s[j+1]:
            j += 1
        t = (si - s[j]) / seg_lens[j] if seg_lens[j] > 0 else 0
        new_pts.append(points[j] * (1 - t) + points[j+1] * t)
    return np.array(new_pts)

def create_curve_mesh_from_points(points, radius=0.1, segments=8):
    """
    从点序列创建管状曲线网格
    """
    n_points = len(points)

    # 计算切线方向
    tangents = []
    for i in range(n_points):
        if i == 0:
            tangent = points[1] - points[0]
        elif i == n_points - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = (points[i + 1] - points[i - 1]) / 2.0
        tangent = tangent / np.linalg.norm(tangent)
        tangents.append(tangent)
    tangents = np.array(tangents)
    up_vec = np.array([0, 0, 1])
    if abs(np.dot(tangents[0], up_vec)) > 0.99:
        up_vec = np.array([0, 1, 0])
    normals = []
    for i in range(n_points):
        if i == 0:
            normal = np.cross(tangents[0], up_vec)
        else:
            normal = np.cross(tangents[i], tangents[i - 1])
            if np.linalg.norm(normal) < 1e-6:
                normal = normals[-1]

        normal = normal / np.linalg.norm(normal)
        binormal = np.cross(tangents[i], normal)
        normals.append(normal)
    normals = np.array(normals)
    # 生成管状网格的顶点
    vertices = []
    triangles = []
    for i in range(n_points):
        for j in range(segments):
            angle = 2 * np.pi * j / segments
            offset = radius * (np.cos(angle) * normals[i] +
                               np.sin(angle) * np.cross(tangents[i], normals[i]))
            vertex = points[i] + offset
            vertices.append(vertex)
    # 生成三角形
    for i in range(n_points - 1):
        for j in range(segments):
            j_next = (j + 1) % segments

            # 当前圆上的顶点索引
            v00 = i * segments + j
            v01 = i * segments + j_next
            v10 = (i + 1) * segments + j
            v11 = (i + 1) * segments + j_next
            # 两个三角形组成四边形
            triangles.append([v00, v10, v01])
            triangles.append([v01, v10, v11])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    return mesh


def calculate_tube_curve_length(mesh, num_sections=100):
    """
    计算管状曲线网格的长度
    通过计算横截面中心点之间的距离
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # 获取所有顶点
    all_vertices = vertices

    pca = PCA(n_components=3)
    pca.fit(all_vertices)

    # 主要方向（曲线的走向）
    main_direction = pca.components_[0]

    # 将顶点投影到主要方向
    projections = np.dot(all_vertices, main_direction)

    # 沿主要方向切片
    min_proj = np.min(projections)
    max_proj = np.max(projections)

    # 创建切片
    slice_positions = np.linspace(min_proj, max_proj, num_sections)

    # 计算每个切片的中心点
    centers = []
    for pos in slice_positions:
        # 找到该切片附近的点
        mask = np.abs(projections - pos) < (max_proj - min_proj) / (num_sections * 2)
        if np.sum(mask) > 0:
            slice_points = all_vertices[mask]
            center = np.mean(slice_points, axis=0)
            centers.append(center)

    # 计算中心点连线长度
    curve_length = 0.0
    for i in range(len(centers) - 1):
        curve_length += np.linalg.norm(centers[i] - centers[i + 1])

    return curve_length


