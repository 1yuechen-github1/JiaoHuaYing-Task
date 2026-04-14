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
from scipy.integrate import quad

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


def get_jhy_w(
    jhy_points,
    cent_list,
    step_mm=1.0,
    vis_list_h=[],
    axiox_list=[],
    oral_scan_center=None,
    return_offsets=False,
):
    """
    Measure keratinized gingiva width.
    """
    sample_axis = cent_list[0]
    # Use centroid of keratinized gingiva points as slice center.
    center = np.mean(jhy_points, axis=0)
    projections = np.dot(jhy_points, sample_axis)
    center_proj = np.dot(center, sample_axis)
    offsets = _build_slice_offsets(center_proj, projections, step_mm)
    slice_positions = [center_proj + off * step_mm for off in offsets]

    tolerance = step_mm / 30.0
    pcd_list = []
    dist_list = []
    slice_points_all = []

    for pos in slice_positions:
        mask = np.abs(projections - pos) <= tolerance
        hit_count = int(np.count_nonzero(mask))
        slice_points = None
        if hit_count > 0:
            slice_points = jhy_points[mask]
            dist = get_len(slice_points, cent_list[1])
            dist_list.append(dist)
            pcd_list.append(get_poin_list(slice_points, [[0, 0, 1]]))
            slice_points_all.append(slice_points)

        else:
            dist_list.append(0)

        pcd = o3d.geometry.PointCloud()
        colors_blue = np.tile([0.7, 0.7, 0.7], (jhy_points.shape[0], 1))
        pcd.points = o3d.utility.Vector3dVector(jhy_points)
        pcd.colors = o3d.utility.Vector3dVector(colors_blue)
        geoms = [pcd]

        if len(slice_points_all) > 0:
            pcd1_all = o3d.geometry.PointCloud()
            all_slice_points = np.vstack(slice_points_all)
            colors_red = np.tile([1, 0, 0], (all_slice_points.shape[0], 1))
            pcd1_all.points = o3d.utility.Vector3dVector(all_slice_points)
            pcd1_all.colors = o3d.utility.Vector3dVector(colors_red)
            geoms.append(pcd1_all)

        if slice_points is not None:
            pcd_cur = o3d.geometry.PointCloud()
            colors_green = np.tile([0, 1, 0], (slice_points.shape[0], 1))
            pcd_cur.points = o3d.utility.Vector3dVector(slice_points)
            pcd_cur.colors = o3d.utility.Vector3dVector(colors_green)
            geoms.append(pcd_cur)

        print(f"[jhy_w] pos={pos:.4f}, hit={hit_count}, accumulated={len(slice_points_all)}")
        vis(geoms, "measure_jhy_w")

    # if len(slice_points_all) > 0:


    if return_offsets:
        return pcd_list, dist_list, offsets
    return pcd_list, dist_list


def get_jhy_h(
    jhy_points,
    cent_list,
    step_mm=1.0,
    vis_list_h=[],
    axiox_list=[],
    oral_scan_center=None,
    is_upper=None,
    return_offsets=False,
):
    """
    Measure keratinized gingiva height.
    """
    sample_axis = cent_list[1]
    center = jhy_points[np.argmax(jhy_points[:, 2])]
    projections = np.dot(jhy_points, sample_axis)
    center_proj = np.dot(center, sample_axis)
    offsets = _build_slice_offsets(center_proj, projections, step_mm)
    slice_positions = [center_proj + off * step_mm for off in offsets]

    tolerance = step_mm / 30.0
    pcd_list = []
    dist_list = []
    slice_points_all = []

    for pos in slice_positions:
        mask = np.abs(projections - pos) <= tolerance
        hit_count = int(np.count_nonzero(mask))
        slice_points = None
        if hit_count > 0:
            slice_points = jhy_points[mask]
            dist = get_len(slice_points, cent_list[0])
            dist_list.append(dist)
            pcd_list.append(get_poin_list(slice_points, [[0, 0, 1]]))
            slice_points_all.append(slice_points)


        else:
            dist_list.append(0)

        
        pcd = o3d.geometry.PointCloud()
        colors_blue = np.tile([0.7, 0.7, 0.7], (jhy_points.shape[0], 1))
        pcd.points = o3d.utility.Vector3dVector(jhy_points)
        pcd.colors = o3d.utility.Vector3dVector(colors_blue)

        geoms = [pcd]
        if len(slice_points_all) > 0:
            pcd1_all = o3d.geometry.PointCloud()
            all_slice_points = np.vstack(slice_points_all)
            colors_red = np.tile([1, 0, 0], (all_slice_points.shape[0], 1))
            pcd1_all.points = o3d.utility.Vector3dVector(all_slice_points)
            pcd1_all.colors = o3d.utility.Vector3dVector(colors_red)
            geoms.append(pcd1_all)

        if slice_points is not None:
            pcd_cur = o3d.geometry.PointCloud()
            colors_green = np.tile([0, 1, 0], (slice_points.shape[0], 1))
            pcd_cur.points = o3d.utility.Vector3dVector(slice_points)
            pcd_cur.colors = o3d.utility.Vector3dVector(colors_green)
            geoms.append(pcd_cur)

        print(f"[jhy_h] pos={pos:.4f}, hit={hit_count}, accumulated={len(slice_points_all)}")
        vis(geoms, "measure_jhy_h")

    # if len(slice_points_all) > 0:


    if return_offsets:
        return pcd_list, dist_list, offsets
    return pcd_list, dist_list




def get_len(points,axis):
    # print('axis',axis)
    radius = 200
    points = np.asarray(points)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # vis([pcd],'file')
    n = len(points)
    G = nx.Graph()
    kdt = cKDTree(points)
    for i, p in enumerate(points):
        idxs = kdt.query_ball_point(p, r=radius)
        for j in idxs:
            if i != j:
                dist = np.linalg.norm(points[i] - points[j])
                G.add_edge(i, j, weight=dist)
    if not nx.has_path(G, 0, n - 1):
        raise ValueError("图不连通，请增大 radius 以完成拟合")
    length = nx.shortest_path_length(G, 0, n - 1, weight='weight')
    # print("length", length)
    return length

# def  get_len(points,axis):
    # print('axis',axis)
    # print('points',points)
    points = points[:, :2]
    x = points[:, 0]
    y = points[:, 1]
    tck, u = splprep([x, y], s=2.0)
    # 定义曲线函数
    def curve_func(t):
        return np.array(splev(t, tck))
    # 曲线长度积分
    def integrand(t):
        dx, dy = splev(t, tck, der=1)
        return np.sqrt(dx**2 + dy**2)
    length, _ = quad(integrand, 0, 1)
    print("曲线长度:", length)
    return length

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


# def interpolate_points_linear(points, num_samples=100):
    # """
    # 线性插值增加点密度。
    # """
    # points = np.asarray(points)
    # diffs = np.diff(points, axis=0)
    # seg_lens = np.linalg.norm(diffs, axis=1)

    # s = np.concatenate([[0], np.cumsum(seg_lens)])
    # s_new = np.linspace(0, s[-1], num_samples)

    # new_pts = []
    # j = 0
    # for si in s_new:
    #     while j < len(seg_lens) - 1 and si > s[j+1]:
    #         j += 1
    #     t = (si - s[j]) / seg_lens[j] if seg_lens[j] > 0 else 0
    #     new_pts.append(points[j] * (1 - t) + points[j+1] * t)
    # return np.array(new_pts)


# def create_curve_mesh_from_points(points, radius=0.1, segments=8):
    # """
    # 从点序列创建管状曲线网格。
    # """
    # n_points = len(points)

    # # 计算切线方向
    # tangents = []
    # for i in range(n_points):
    #     if i == 0:
    #         tangent = points[1] - points[0]
    #     elif i == n_points - 1:
    #         tangent = points[-1] - points[-2]
    #     else:
    #         tangent = (points[i + 1] - points[i - 1]) / 2.0
    #     tangent = tangent / np.linalg.norm(tangent)
    #     tangents.append(tangent)
    # tangents = np.array(tangents)
    # up_vec = np.array([0, 0, 1])
    # if abs(np.dot(tangents[0], up_vec)) > 0.99:
    #     up_vec = np.array([0, 1, 0])
    # normals = []
    # for i in range(n_points):
    #     if i == 0:
    #         normal = np.cross(tangents[0], up_vec)
    #     else:
    #         normal = np.cross(tangents[i], tangents[i - 1])
    #         if np.linalg.norm(normal) < 1e-6:
    #             normal = normals[-1]

    #     normal = normal / np.linalg.norm(normal)
    #     binormal = np.cross(tangents[i], normal)
    #     normals.append(normal)
    # normals = np.array(normals)
    # # 生成管状网格顶点
    # vertices = []
    # triangles = []
    # for i in range(n_points):
    #     for j in range(segments):
    #         angle = 2 * np.pi * j / segments
    #         offset = radius * (np.cos(angle) * normals[i] +
    #                            np.sin(angle) * np.cross(tangents[i], normals[i]))
    #         vertex = points[i] + offset
    #         vertices.append(vertex)
    # # 生成三角形面片
    # for i in range(n_points - 1):
    #     for j in range(segments):
    #         j_next = (j + 1) % segments

    #         # 当前圆环上的顶点索引
    #         v00 = i * segments + j
    #         v01 = i * segments + j_next
    #         v10 = (i + 1) * segments + j
    #         v11 = (i + 1) * segments + j_next
    #         # 两个三角形组成一个四边面
    #         triangles.append([v00, v10, v01])
    #         triangles.append([v01, v10, v11])
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    # mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    # return mesh


# def calculate_tube_curve_length(mesh, num_sections=100):
    """
    # 计算管状曲线网格的长度。
    # 通过截面中心点连线长度近似曲线长度。
    # """
    # vertices = np.asarray(mesh.vertices)
    # triangles = np.asarray(mesh.triangles)

    # # 获取所有顶点
    # all_vertices = vertices

    # pca = PCA(n_components=3)
    # pca.fit(all_vertices)

    # # 主要方向（曲线走向）
    # main_direction = pca.components_[0]

    # # 将顶点投影到主要方向
    # projections = np.dot(all_vertices, main_direction)

    # # 沿主要方向切片
    # min_proj = np.min(projections)
    # max_proj = np.max(projections)

    # # 创建切片位置
    # slice_positions = np.linspace(min_proj, max_proj, num_sections)

    # # 计算每个切片的中心点
    # centers = []
    # for pos in slice_positions:
    #     # 找到切片附近点
    #     mask = np.abs(projections - pos) < (max_proj - min_proj) / (num_sections * 2)
    #     if np.sum(mask) > 0:
    #         slice_points = all_vertices[mask]
    #         center = np.mean(slice_points, axis=0)
    #         centers.append(center)

    # # 累加中心点连线长度
    # curve_length = 0.0
    # for i in range(len(centers) - 1):
    #     curve_length += np.linalg.norm(centers[i] - centers[i + 1])

    # return curve_length


def label_w(offset):
    if offset == 0:
        return "中央(z最大)"
    if offset < 0:
        return f"近中{abs(offset)}mm"
    return f"远中{offset}mm"


def label_h(offset, is_upper):
    if offset == 0:
        return "中央(z最大)"

    is_upper_case = str(is_upper).lower()
    # 上颌统一为偏颊侧；下颌按舌侧/颊侧区分
    if is_upper_case in {"upper", "up", "u", "maxilla"}:
        return f"偏颊侧{abs(offset)}mm"

    if offset > 0:
        return f"偏舌侧{offset}mm"
    return f"偏颊侧{abs(offset)}mm"

