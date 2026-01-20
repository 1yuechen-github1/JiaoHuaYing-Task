import os

import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
from collections import Counter
import matplotlib.pyplot as plt


# def vis(pcd, file):
#     o3d.visualization.draw_geometries([pcd], 
#                              window_name=f"Point Cloud: {file}",
#                              width=800, 
#                              height=600)

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
    
    # X轴 (红色) Y轴 (绿色) Z轴 (蓝色)
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


# def get_jhy_h(jhy_points,jhy_center,cent_list, step_mm, file):
#     sample_axis = cent_list[1]  # 这是采样方向
#     projections = np.dot(jhy_points, sample_axis)
#     proj_min = np.min(projections)
#     proj_max = np.max(projections)
#     sample_positions = np.arange(proj_min, proj_max, step_mm)
#     slices = []
#     pcd_list = []
#     dist_list = []
#     for pos in sample_positions:
#         # 找到在当前位置附近的点
#         tolerance = step_mm / 50.0
#         # print(tolerance, step_mm)
#         mask = np.abs(projections - pos) <= tolerance
#         if np.any(mask):
#             slice_points = jhy_points[mask]
#             dist = get_len(slice_points,sample_axis)
#             dist_list.append(dist)
#             pcd = get_poin_list(slice_points,[[0, 0, 1]])
#             pcd_list.append(pcd)
#     return  pcd_list,dist_list

# def get_jhy_w(jhy_points,jhy_center,cent_list, step_mm, file):
#     sample_axis = cent_list[0]  # 这是采样方向
#     projections = np.dot(jhy_points, sample_axis)
#     proj_min = np.min(projections)
#     proj_max = np.max(projections)
#     sample_positions = np.arange(proj_min, proj_max, step_mm)
#     slices = []
#     pcd_list = []
#     dist_list = []
#     for pos in sample_positions:
#         # 找到在当前位置附近的点
#         tolerance = step_mm / 50.0
#         mask = np.abs(projections - pos) <= tolerance
#         if np.any(mask):
#             slice_points = jhy_points[mask]
#             dist = get_len(slice_points,sample_axis)
#             dist_list.append(dist)
#             pcd = get_poin_list(slice_points,[[0, 0, 1]])
#             pcd_list.append(pcd)
#     return  pcd_list,dist_list

# X轴 (黑色) Y轴 (绿色) Z轴 (蓝色)
def get_jhy_w(jhy_points, cent_list, step_mm=1.0):
    """
    只在中心及上下 1mm 位置做 3 个切片
    """
    sample_axis = cent_list[0]          # 采样方向（单位向量）
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
            dist = get_len(slice_points, sample_axis)
            dist_list.append(dist)
            pcd = get_poin_list(slice_points, [[0, 0, 1]])
            pcd_list.append(pcd)
        else:
            dist_list.append(0)
    return pcd_list, dist_list


def get_jhy_h(jhy_points, cent_list, step_mm=1.0):
    """
    只在中心及上下 1mm 位置做 3 个切片
    """
    sample_axis = cent_list[1]          # 采样方向（单位向量）
    center = np.mean(jhy_points, axis=0)
    # 所有点在采样轴上的投影
    projections = np.dot(jhy_points, sample_axis)
    center_proj = np.dot(center, sample_axis)
    #
    # index_list = [0, -1, 1, -2, 2, -3, 3]
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
            dist = get_len(slice_points, sample_axis)
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
    return np.sum(valid)


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

def draw_juxin():
    return
    # points_np = np.asarray(pcd.points )
    # min_bound = points_np.min(axis=0)  
    # max_bound = points_np.max(axis=0)  
    # step = 0.5
    # xs = np.arange(min_bound[0], max_bound[0]+step, step)
    # ys = np.arange(min_bound[1], max_bound[1]+step, step)
    # zs = np.arange(min_bound[2], max_bound[2]+step, step)
    # grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
    # cube_pcd = o3d.geometry.PointCloud()
    # cube_pcd.points = o3d.utility.Vector3dVector(grid)
    # # 给点云一个颜色，例如绿色
    # cube_colors = np.tile([0, 1, 0], (grid.shape[0], 1))
    # cube_pcd.colors = o3d.utility.Vector3dVector(cube_colors)