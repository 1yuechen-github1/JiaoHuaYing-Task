import os
import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import Delaunay, cKDTree
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import csv


DEBUG_VIS_GET_LEN_CLUSTERS = True

TARGET_OFFSETS = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]


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
    使用 HSV 颜色空间过滤，保留非红色点。
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
    
    # 提取 HSV 分量
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
    # 映射为新的标签（0/1）
    label_mapping = {top_two_labels[0]: 0, top_two_labels[1]: 1}
    new_labels = np.array([label_mapping[label] for label in remaining_labels])
    colors = plt.get_cmap("tab20")(new_labels / 1)  # 仅有 0 和 1 两类
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
    # print('poin_list', len(poin_list))
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


def fit_measure_arc(points, sort_axis, samples=120):
    """对当前切片的原始测量点拟合并采样连续弧线。"""
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 4:
        return points

    sort_axis = np.asarray(sort_axis, dtype=float)
    ordered = points[np.argsort(points @ sort_axis)]
    # 仅使用切片中的原始点；s=0 使曲线穿过这些点，不向切片外延伸。
    try:
        tck, _ = splprep(ordered.T, s=0.0, k=min(3, len(ordered) - 1))
        return np.asarray(splev(np.linspace(0.0, 1.0, samples), tck)).T
    except (ValueError, TypeError):
        return ordered


def create_coordinate_frame(center, x_axis, y_axis, z_axis, scale=1.0):
    """
    创建自定义坐标系。
    
    参数:
        center: 坐标系原点
        x_axis, y_axis, z_axis: 三个轴方向向量
        scale: 坐标轴长度缩放
    
    返回:
        coordinate_frame: 包含三条轴线的 Open3D 几何体
    """
    center = np.array(center)
    x_axis = np.array(x_axis) * scale
    y_axis = np.array(y_axis) * scale
    z_axis = np.array(z_axis) * scale
    
    # 创建坐标轴线段
    axes = []
    
    # X 轴(黑色) Y 轴(绿色) Z 轴(蓝色)
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



# X 轴(黑色) Y 轴(绿色) Z 轴(蓝色)
def _build_slice_offsets(center_proj, projections, step_mm):
    min_proj = float(np.min(projections))
    max_proj = float(np.max(projections))
    max_neg = int(np.floor((center_proj - min_proj) / step_mm))
    max_pos = int(np.floor((max_proj - center_proj) / step_mm))

    offsets = [0]
    max_step = max(max_neg, max_pos)
    for i in range(1, max_step + 1):
        if i <= max_neg:
            offsets.append(-i)
        if i <= max_pos:
            offsets.append(i)
    return offsets


# 最开始代码

# def get_jhy_w(
#     jhy_points,
#     jhy_colors,
#     cent_list,
#     step_mm=1.0,
#     vis_list_h=[],
#     axiox_list=[],
#     oral_scan_center=None,
#     return_offsets=False,
# ):
#     """
#     Measure keratinized gingiva width.
#     """
#     sample_axis = cent_list[0]
#     # Use centroid of keratinized gingiva points as slice center.
#     center = np.mean(jhy_points, axis=0)
#     projections = np.dot(jhy_points, sample_axis)
#     center_proj = np.dot(center, sample_axis)
#     offsets = _build_slice_offsets(center_proj, projections, step_mm)
#     slice_positions = [center_proj + off * step_mm for off in offsets]

#     # 同一批切片点既用于测量，也用于导出。适当加宽采样带，避免弧线断续。
#     tolerance = step_mm / 30.0
#     pcd_list = []
#     dist_list = []
#     slice_points_all = []

#     for pos in slice_positions:
#         mask = np.abs(projections - pos) <= tolerance
#         hit_count = int(np.count_nonzero(mask))
#         slice_points = None
#         if hit_count > 0:
#             slice_points = jhy_points[mask]
#             dist = get_len(slice_points, cent_list[1])
#             dist_list.append(dist)
#             # 导出最短测量路径的加密点；该路径与 get_len 的弧长计算完全一致。
#             # pcd_list.append(get_poin_list(fit_measure_arc(slice_points, cent_list[1]), [[0, 0, 1]]))
#             pcd_list.append(get_poin_list(slice_points, [[0, 0, 1]]))  
#             slice_points_all.append(slice_points)

#         else:
#             dist_list.append(0)

#         pcd = o3d.geometry.PointCloud()
#         colors_blue = jhy_colors
#         pcd.points = o3d.utility.Vector3dVector(jhy_points)
#         pcd.colors = o3d.utility.Vector3dVector(colors_blue)
#         # geoms = [pcd]
#         geoms = []
#         if len(slice_points_all) > 0:
#             pcd1_all = o3d.geometry.PointCloud()
#             all_slice_points = np.vstack(slice_points_all)
#             colors_red = np.tile([1, 0, 0], (all_slice_points.shape[0], 1))
#             pcd1_all.points = o3d.utility.Vector3dVector(all_slice_points)
#             pcd1_all.colors = o3d.utility.Vector3dVector(colors_red)
#             geoms.append(pcd1_all)

#         if slice_points is not None:
#             pcd_cur = o3d.geometry.PointCloud()
#             colors_green = np.tile([0, 1, 0], (slice_points.shape[0], 1))
#             pcd_cur.points = o3d.utility.Vector3dVector(slice_points)
#             pcd_cur.colors = o3d.utility.Vector3dVector(colors_green)
#             geoms.append(pcd_cur)


#     if return_offsets:
#         return pcd_list, dist_list, offsets
#     return pcd_list, dist_list



def get_jhy_w(
    jhy_points,
    jhy_colors,
    cent_list,
    step_mm=1.0,
    vis_list_h=None,
    axiox_list=None,
    oral_scan_center=None,
    return_offsets=False,
    keep_mm=2.0,
):
    """
    按原始方法生成宽度弧线，然后进行颊侧球形采样。

    原始弧线生成方式：
        沿 cent_list[0] 方向进行切片。

    每条弧线处理方式：
        1. 得到完整切片弧线；
        2. 找到距离口扫中心最远的点；
        3. 以该点为球心；
        4. 半径 keep_mm 球形采样；
        5. 保留当前弧线中位于球内的点。
    """

    if vis_list_h is None:
        vis_list_h = []

    if axiox_list is None:
        axiox_list = []

    jhy_points = np.asarray(
        jhy_points,
        dtype=float,
    )

    jhy_colors = np.asarray(
        jhy_colors,
        dtype=float,
    )

    if jhy_points.ndim != 2 or jhy_points.shape[0] < 2:
        if return_offsets:
            return [], [], []

        return [], []

    if oral_scan_center is None:
        raise ValueError(
            "oral_scan_center 不能为空"
        )

    oral_scan_center = np.asarray(
        oral_scan_center,
        dtype=float,
    ).reshape(-1, 3)

    if oral_scan_center.shape[0] == 0:
        raise ValueError(
            "oral_scan_center 不能为空"
        )

    oral_center = oral_scan_center[0]

    # =========================================================
    # 1. 按照原始方法，沿 axiox1 方向生成弧线
    # =========================================================
    sample_axis = np.asarray(
        cent_list[0],
        dtype=float,
    )

    sample_axis_norm = np.linalg.norm(
        sample_axis
    )

    if sample_axis_norm < 1e-12:
        if return_offsets:
            return [], [], []

        return [], []

    sample_axis = (
        sample_axis / sample_axis_norm
    )

    projections = np.dot(
        jhy_points,
        sample_axis,
    )

    center = np.mean(
        jhy_points,
        axis=0,
    )

    center_proj = np.dot(
        center,
        sample_axis,
    )

    offsets = _build_slice_offsets(
        center_proj,
        projections,
        step_mm,
    )

    slice_positions = [
        center_proj + offset * step_mm
        for offset in offsets
    ]

    # 与原始方法保持一致
    tolerance = step_mm / 30.0

    pcd_list = []
    dist_list = []
    valid_offsets = []

    # =========================================================
    # 2. 逐条弧线处理
    # =========================================================
    for offset, position in zip(
        offsets,
        slice_positions,
    ):
        # 原始切片方式
        mask = (
            np.abs(projections - position)
            <= tolerance
        )

        slice_points = jhy_points[mask]

        if slice_points.shape[0] < 2:
            continue

        # =====================================================
        # 3. 找到距离口扫中心最远的弧线上点
        # =====================================================
        distances_to_oral = np.linalg.norm(
            slice_points - oral_center,
            axis=1,
        )

        farthest_index = np.argmax(
            distances_to_oral
        )

        sphere_center = slice_points[
            farthest_index
        ]

        # =====================================================
        # 4. 以该点为球心，半径 2 mm 球形采样
        # =====================================================
        distances_to_sphere_center = np.linalg.norm(
            slice_points - sphere_center,
            axis=1,
        )

        keep_mask = (
            distances_to_sphere_center
            <= keep_mm
        )

        selected_points = slice_points[
            keep_mask
        ]

        if selected_points.shape[0] < 2:
            continue

        # =====================================================
        # 5. 计算保留弧线长度
        # =====================================================
        selected_dist = get_len(
            selected_points,
            cent_list[1],
        )

        if selected_dist <= 0:
            continue

        # =====================================================
        # 6. 保存当前弧线
        # =====================================================
        arc_pcd = get_poin_list(
            selected_points,
            [[1.0, 0.0, 0.0]],
        )

        pcd_list.append(arc_pcd)
        dist_list.append(float(selected_dist))
        valid_offsets.append(offset)


    if return_offsets:
        return (
            pcd_list,
            dist_list,
            valid_offsets,
        )

    
    return pcd_list, dist_list






def _resample_curve(points, sample_count):
    """
    将一条弧线按照弧长重新采样为固定点数。
    """
    points = np.asarray(points, dtype=float)

    if points.shape[0] < 2:
        return points

    segment_lengths = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1,
    )

    cumulative_length = np.concatenate(
        [
            np.array([0.0]),
            np.cumsum(segment_lengths),
        ]
    )

    total_length = cumulative_length[-1]

    if total_length <= 1e-12:
        return np.repeat(
            points[:1],
            sample_count,
            axis=0,
        )

    target_lengths = np.linspace(
        0.0,
        total_length,
        sample_count,
    )

    new_points = []

    for target in target_lengths:
        index = np.searchsorted(
            cumulative_length,
            target,
            side="right",
        ) - 1

        index = max(
            0,
            min(index, len(points) - 2),
        )

        local_length = (
            cumulative_length[index + 1]
            - cumulative_length[index]
        )

        if local_length <= 1e-12:
            ratio = 0.0
        else:
            ratio = (
                target
                - cumulative_length[index]
            ) / local_length

        point = (
            points[index]
            + ratio
            * (points[index + 1] - points[index])
        )

        new_points.append(point)

    return np.asarray(new_points)


def oversampling(
    pcd_list,
    dist_list,
    offsets,
    oversample_num=2,
    sample_points=100,
):
    """
    在相邻两条弧线之间插入过采样弧线。

    参数
    ----------
    pcd_list:
        原始弧线 PointCloud 列表。

    dist_list:
        原始弧长列表。

    offsets:
        原始弧线 offset 列表。

    oversample_num:
        相邻弧线之间插入多少条弧线。
        例如：
            1：插入 1 条
            2：插入 2 条
            4：插入 4 条

    sample_points:
        每条弧线重新采样的点数。

    返回
    ----------
    new_pcd_list:
        包含原始弧线和新增弧线的列表。

    new_dist_list:
        对应弧长列表。

    new_offsets:
        对应 offset 列表。
    """
    if len(pcd_list) < 2:
        return pcd_list, dist_list, offsets

    if len(pcd_list) != len(offsets):
        raise ValueError(
            "pcd_list 和 offsets 长度不一致"
        )

    oversample_num = max(
        int(oversample_num),
        0,
    )

    sample_points = max(
        int(sample_points),
        2,
    )

    # 按 offset 排序，确保相邻弧线顺序正确
    order = np.argsort(offsets)

    pcd_list = [
        pcd_list[i]
        for i in order
    ]

    offsets = [
        offsets[i]
        for i in order
    ]

    dist_list = [
        dist_list[i]
        for i in order
    ]

    # 将所有弧线重新采样为相同点数
    curves = []

    for pcd in pcd_list:
        points = np.asarray(
            pcd.points,
            dtype=float,
        )

        if points.shape[0] < 2:
            curves.append(points)
        else:
            curves.append(
                _resample_curve(
                    points,
                    sample_points,
                )
            )

    new_pcd_list = []
    new_dist_list = []
    new_offsets = []

    for i in range(len(curves) - 1):
        curve0 = curves[i]
        curve1 = curves[i + 1]

        # 保存当前原始弧线
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(
            curve0
        )
        original_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(
                [1.0, 0.0, 0.0],
                (curve0.shape[0], 1),
            )
        )

        new_pcd_list.append(original_pcd)
        new_dist_list.append(float(dist_list[i]))
        new_offsets.append(float(offsets[i]))

        # 在两条弧线之间插值
        for j in range(1, oversample_num + 1):
            ratio = j / (
                oversample_num + 1
            )

            interpolated_curve = (
                (1.0 - ratio) * curve0
                + ratio * curve1
            )

            interpolated_pcd = (
                o3d.geometry.PointCloud()
            )

            interpolated_pcd.points = (
                o3d.utility.Vector3dVector(
                    interpolated_curve
                )
            )

            interpolated_pcd.colors = (
                o3d.utility.Vector3dVector(
                    np.tile(
                        [1.0, 0.5, 0.0],
                        (
                            interpolated_curve.shape[0],
                            1,
                        ),
                    )
                )
            )

            interpolated_offset = (
                (1.0 - ratio) * offsets[i]
                + ratio * offsets[i + 1]
            )

            interpolated_dist = (
                (1.0 - ratio) * dist_list[i]
                + ratio * dist_list[i + 1]
            )

            new_pcd_list.append(
                interpolated_pcd
            )

            new_dist_list.append(
                float(interpolated_dist)
            )

            new_offsets.append(
                float(interpolated_offset)
            )

    # 保存最后一条原始弧线
    last_curve = curves[-1]

    last_pcd = o3d.geometry.PointCloud()
    last_pcd.points = o3d.utility.Vector3dVector(
        last_curve
    )
    last_pcd.colors = o3d.utility.Vector3dVector(
        np.tile(
            [1.0, 0.0, 0.0],
            (last_curve.shape[0], 1),
        )
    )

    new_pcd_list.append(last_pcd)
    new_dist_list.append(float(dist_list[-1]))
    new_offsets.append(float(offsets[-1]))

    return (
        new_pcd_list,
        new_dist_list,
        new_offsets,
    )



def get_len(points,axis):
    # print('len(points):',len(points))
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # vis([pcd], '')

    radius = 400
    points = np.asarray(points)
    if points.shape[0] < 2:
        return 0.0

    n = len(points)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    kdt = cKDTree(points)
    for i, point in enumerate(points):
        for j in kdt.query_ball_point(point, r=radius):
            if i != j:
                graph.add_edge(i, j, weight=np.linalg.norm(points[i] - points[j]))

    if not nx.has_path(graph, 0, n - 1):
        return 0.0

    length = nx.shortest_path_length(graph, 0, n - 1, weight='weight')
    # print("length", length)
    return float(length)


def save_to_txt(pcd2, pcd_list, output, file, scalar, status, offsets, target_offsets):
    points_array = np.asarray(pcd2.points)
    colors_array = np.asarray(pcd2.colors)
    colors_array = (colors_array * 255).astype(np.uint8)
    all_points = []
    all_colors = []
    selected_pcds = [pcd for offset, pcd in zip(offsets, pcd_list) if offset in target_offsets]
    has_colors = all(p.has_colors() for p in selected_pcds if p is not None)
    for pcd in selected_pcds:
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
    os.makedirs(os.path.join(output,status), exist_ok=True)
    np.savetxt(f"{output}\\{status}\\{file}",save_array,fmt="%.6f %.6f %.6f %.6f %.6f %.6f %.6f")




def label_w(offset):
    if offset == 0:
        return "中央(z最大)"
    if offset < 0:
        return f"近中{abs(offset/2)}mm"
    return f"远中{offset/2}mm"


def label_h(offset, is_upper):
    if offset == 0:
        return "中央(z最大)"

    is_upper_case = str(is_upper).lower()
    # 上颌统一为偏颊侧；下颌按舌侧/颊侧区分
    if is_upper_case in {"upper", "up", "u", "maxilla"}:
        return f"偏颊侧{abs(offset/2)}mm"

    if offset > 0:
        return f"偏舌侧{offset/2}mm"
    return f"偏颊侧{abs(offset/2)}mm"


def append_measure_csv(csv_path, file, offsets, dists, label_h,is_upper):
    dist_by_offset = dict(zip(offsets, dists))
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    if is_upper != None:
        is_upper_case = str(is_upper).lower()
        # 长度：上颌只导出颊侧（offset >= 0）；下颌导出舌侧/颊侧两侧。
        target_offsets = (
            [off for off in TARGET_OFFSETS if off >= 0]
            if is_upper_case in {"upper", "up", "u", "maxilla"}
            else TARGET_OFFSETS
        )
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["file"] + [label_h(off, is_upper) for off in target_offsets])
            writer.writerow([file] + [dist_by_offset.get(off, 0) for off in target_offsets])

    else:
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["file"] + [f"{label_h(off)}" for off in TARGET_OFFSETS])
            writer.writerow([file] + [dist_by_offset.get(off, 0) for off in TARGET_OFFSETS])


def split_buccal_lingual_points(
    slice_points,
    axiox2,
    buccal_keep_mm=2.0,
):
    """
    根据 axiox2 将点分为颊侧和舌侧。

    axiox2 必须定义为：
        颊侧 -> 舌侧

    返回：
        buccal_points       颊侧全部点
        buccal_2mm_points   靠近颊侧的 2 mm 点
        lingual_points      舌侧点
    """
    slice_points = np.asarray(slice_points, dtype=float)

    if slice_points.shape[0] == 0:
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 3)),
        )

    axiox2 = np.asarray(axiox2, dtype=float)
    axiox2 = axiox2 / (np.linalg.norm(axiox2) + 1e-12)

    # 点在颊舌方向上的投影
    proj = np.dot(slice_points, axiox2)

    # 因为 axiox2 是颊侧指向舌侧，
    # 投影最小的位置就是颊侧边界
    buccal_proj = np.min(proj)

    # 颊侧边界向舌侧方向 2 mm 的范围
    buccal_mask = proj <= buccal_proj + buccal_keep_mm

    buccal_2mm_points = slice_points[buccal_mask]
    lingual_points = slice_points[~buccal_mask]

    # 颊侧全部点
    buccal_points = slice_points[proj <= np.median(proj)]

    return buccal_points, buccal_2mm_points, lingual_points

def create_side_visualization(
    all_points,
    buccal_points,
    buccal_2mm_points,
    lingual_points,
    save_path,
):
    """
    生成颊侧/舌侧可视化点云。

    颜色：
        灰色：原始点
        红色：颊侧 2 mm 保留区域
        蓝色：舌侧区域
        绿色：当前测量点
    """
    geometries = []

    all_points = np.asarray(all_points)
    buccal_points = np.asarray(buccal_points)
    buccal_2mm_points = np.asarray(buccal_2mm_points)
    lingual_points = np.asarray(lingual_points)

    if all_points.shape[0] > 0:
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(all_points)
        pcd_all.colors = o3d.utility.Vector3dVector(
            np.tile([0.65, 0.65, 0.65], (all_points.shape[0], 1))
        )
        geometries.append(pcd_all)

    if lingual_points.shape[0] > 0:
        pcd_lingual = o3d.geometry.PointCloud()
        pcd_lingual.points = o3d.utility.Vector3dVector(lingual_points)
        pcd_lingual.colors = o3d.utility.Vector3dVector(
            np.tile([0.1, 0.3, 1.0], (lingual_points.shape[0], 1))
        )
        geometries.append(pcd_lingual)

    if buccal_2mm_points.shape[0] > 0:
        pcd_buccal = o3d.geometry.PointCloud()
        pcd_buccal.points = o3d.utility.Vector3dVector(buccal_2mm_points)
        pcd_buccal.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.05, 0.05], (buccal_2mm_points.shape[0], 1))
        )
        geometries.append(pcd_buccal)

    if save_path:
        o3d.io.write_point_cloud(save_path, sum(geometries[1:], geometries[0]))

    return geometries