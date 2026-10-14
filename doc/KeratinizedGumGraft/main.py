import argparse
import copy
import os

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from pcd_to_mesh import PointCloudMeshConverter


from utils import (
    create_coordinate_frame,
    filt_rpoin_hsv,
    use_dbscan,
    get_poin_list,
    get_jhy_w,
    save_to_txt,
    append_measure_csv,
    label_w,
    oversampling
)


# python code\doc\KeratinizedGumGraft\main.py --keratinized_gingiva Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\pointcept-加上下颌位置信息-法向量-边界做数据增强\result-txt-test --rotake_oral_scan Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\test\rotake_oral_scan --output Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\pointcept-加上下颌位置信息-法向量-边界做数据增强\output1

TARGET_OFFSETS = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]

SLICE_STEP_MM = 0.25

# 从颊侧边界向舌侧保留 2 mm
BUCCAL_KEEP_MM = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="颊侧 2 mm 宽度测量"
    )

    parser.add_argument(
        "--keratinized_gingiva",
        required=True,
        type=str,
        help="角化龈点云目录",
    )

    parser.add_argument(
        "--rotake_oral_scan",
        required=True,
        type=str,
        help="口扫点云目录",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="输出目录",
    )

    return parser.parse_args()


def read_original_oral_scan(file_path):
    """
    读取原始口扫，同时保留原始颜色。

    支持：
        .ply
        .txt: x y z [r g b ...]
    """
    file_path = str(file_path)

    if file_path.lower().endswith(".ply"):
        pcd = o3d.io.read_point_cloud(file_path)

        if len(pcd.points) == 0:
            raise ValueError(f"原始口扫为空: {file_path}")

        # 如果 PLY 没有颜色，设置为灰色
        if not pcd.has_colors():
            colors = np.tile(
                [0.70, 0.70, 0.70],
                (len(pcd.points), 1),
            )
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    data = np.loadtxt(file_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            f"口扫文件至少需要 xyz 三列: {file_path}"
        )

    points = data[:, :3].astype(float)

    if data.shape[1] >= 6:
        colors = data[:, 3:6].astype(float)

        if np.max(colors) > 1.0:
            colors = colors / 255.0

        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.tile(
            [0.70, 0.70, 0.70],
            (len(points), 1),
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def combine_width_measurement_points(pcd_list_w):
    """
    合并 get_jhy_w / oversampling 返回的全部宽度测量点。
    """
    point_arrays = []

    for pcd in pcd_list_w:
        if pcd is None:
            continue

        points = np.asarray(
            pcd.points,
            dtype=np.float64,
        )

        if points.ndim == 2 and points.shape[0] > 0:
            point_arrays.append(points[:, :3])

    if not point_arrays:
        return np.empty((0, 3), dtype=np.float64)

    points = np.concatenate(point_arrays, axis=0)

    # 去除异常值
    points = points[
        np.all(np.isfinite(points), axis=1)
    ]

    if points.shape[0] == 0:
        return points

    # 去除重复点；保留 6 位小数一般不会影响毫米级数据
    _, unique_indices = np.unique(
        np.round(points, decimals=6),
        axis=0,
        return_index=True,
    )

    return points[np.sort(unique_indices)]


def estimate_point_spacing(points):
    """
    估算点云平均点间距。
    """
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] < 2:
        return 0.05

    tree = cKDTree(points)

    distances, _ = tree.query(
        points,
        k=min(2, len(points)),
    )

    if distances.ndim == 1:
        return 0.05

    nearest_distances = distances[:, 1]
    nearest_distances = nearest_distances[
        np.isfinite(nearest_distances)
        & (nearest_distances > 1e-8)
    ]

    if nearest_distances.shape[0] == 0:
        return 0.05

    return float(np.median(nearest_distances))


def get_surface_boundary(
    points,
    labels,
    target_label=1,
    k=12,
):
    """
    根据邻域内是否同时存在目标点和非目标点提取边界。

    points:
        N x 3 点坐标。

    labels:
        N 个标签，测量区域为 1，其他区域为 0。
    """
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)

    if points.shape[0] != labels.shape[0]:
        raise ValueError("points 和 labels 数量不一致")

    target_mask = labels == target_label

    if not np.any(target_mask):
        return np.empty((0, 3), dtype=np.float64)

    if np.all(target_mask):
        # 全部都是目标点时，没有外部邻域可用于判断边界
        return points.copy()

    tree = cKDTree(points)

    query_k = min(k + 1, len(points))

    _, neighbor_indices = tree.query(
        points,
        k=query_k,
    )

    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices.reshape(-1, 1)

    # 第一列一般是点自身
    neighbor_indices = neighbor_indices[:, 1:]

    boundary_indices = []

    for index in np.where(target_mask)[0]:
        neighbors = neighbor_indices[index]

        if neighbors.shape[0] == 0:
            continue

        if np.any(~target_mask[neighbors]):
            boundary_indices.append(index)

    if not boundary_indices:
        return np.empty((0, 3), dtype=np.float64)

    return points[
        np.asarray(boundary_indices, dtype=int)
    ]


def extract_width_area_boundary(
    reference_points,
    width_points,
    area_distance_threshold=None,
    boundary_k=12,
):
    """
    将宽度测量点映射到角化龈参考点云，然后提取测量区域边界。

    reference_points:
        第二段代码中的 jhy_points。

    width_points:
        pcd_list_w 合并后的测量点。

    area_distance_threshold:
        判定参考点是否属于宽度测量区域的距离阈值。
        None 时自动按点间距计算。
    """
    reference_points = np.asarray(
        reference_points,
        dtype=np.float64,
    )

    width_points = np.asarray(
        width_points,
        dtype=np.float64,
    )

    if reference_points.shape[0] == 0:
        return np.empty((0, 3)), np.zeros(0, dtype=bool)

    if width_points.shape[0] == 0:
        return (
            np.empty((0, 3)),
            np.zeros(len(reference_points), dtype=bool),
        )

    spacing = estimate_point_spacing(reference_points)

    if area_distance_threshold is None:
        # 最小值防止阈值太小导致测量区域断裂
        area_distance_threshold = max(
            spacing * 2.5,
            0.08,
        )

    width_tree = cKDTree(width_points)

    distances, _ = width_tree.query(
        reference_points,
        k=1,
    )

    width_area_mask = (
        distances <= area_distance_threshold
    )

    labels = width_area_mask.astype(np.int32)

    boundary_points = get_surface_boundary(
        reference_points,
        labels,
        target_label=1,
        k=boundary_k,
    )

    return boundary_points, width_area_mask


def color_boundary_on_original_scan(
    oral_scan_file,
    boundary_points,
    output_file,
    boundary_color=(1.0, 0.0, 0.0),
    mapping_threshold=None,
    line_radius=None,
):
    """
    把边界叠加到原始口扫。

    原始口扫颜色保持不变，只修改边界附近点的颜色。

    mapping_threshold:
        测量边界到原始口扫的最大允许距离。

    line_radius:
        边界线加粗半径。若为 0，则只染最近邻点。
    """
    oral_pcd = read_original_oral_scan(
        oral_scan_file
    )

    oral_points = np.asarray(
        oral_pcd.points,
        dtype=np.float64,
    )

    oral_colors = np.asarray(
        oral_pcd.colors,
        dtype=np.float64,
    ).copy()

    boundary_points = np.asarray(
        boundary_points,
        dtype=np.float64,
    )

    if boundary_points.shape[0] == 0:
        raise ValueError("没有提取到宽度测量区域边界")

    oral_spacing = estimate_point_spacing(
        oral_points
    )

    if mapping_threshold is None:
        mapping_threshold = max(
            oral_spacing * 3.0,
            0.10,
        )

    if line_radius is None:
        # 稍微加粗边界，否则高密度口扫中只有一个像素点宽
        line_radius = max(
            oral_spacing * 1.5,
            0.05,
        )

    oral_tree = cKDTree(oral_points)

    distances, nearest_indices = oral_tree.query(
        boundary_points,
        k=1,
    )

    valid_mask = (
        np.isfinite(distances)
        & (distances <= mapping_threshold)
    )

    valid_boundary_points = boundary_points[
        valid_mask
    ]

    valid_nearest_indices = nearest_indices[
        valid_mask
    ]

    if valid_boundary_points.shape[0] == 0:
        raise ValueError(
            "边界点无法映射到原始口扫。"
            "请增大 mapping_threshold，"
            "或检查两份点云是否位于同一坐标系。"
        )

    oral_boundary_mask = np.zeros(
        len(oral_points),
        dtype=bool,
    )

    if line_radius > 0:
        # 将边界附近一小圈口扫点着色，使边界清楚可见
        neighbor_lists = oral_tree.query_ball_point(
            valid_boundary_points,
            r=line_radius,
        )

        for indices in neighbor_lists:
            oral_boundary_mask[indices] = True
    else:
        oral_boundary_mask[
            valid_nearest_indices
        ] = True

    oral_colors[oral_boundary_mask] = np.asarray(
        boundary_color,
        dtype=float,
    )

    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(
        oral_points
    )
    result_pcd.colors = o3d.utility.Vector3dVector(
        np.clip(oral_colors, 0.0, 1.0)
    )

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # success = o3d.io.write_point_cloud(
    #     output_file,
    #     result_pcd,
    #     write_ascii=False,
    #     compressed=False,
    # )

    # if not success:
    #     raise IOError(f"保存失败: {output_file}")

    print("宽度区域边界点数:", len(boundary_points))
    print("映射到口扫的边界点数:", len(valid_boundary_points))
    print("口扫着色点数:", int(np.sum(oral_boundary_mask)))
    print("边界可视化已保存:", output_file)

    return result_pcd, oral_boundary_mask


# def visualize_width_boundary_on_oral_scan(
#     oral_scan_file,
#     jhy_points,
#     pcd_list_w,
#     output_file,
#     area_distance_threshold=None,
#     mapping_threshold=None,
#     boundary_k=12,
#     line_radius=None,
# ):
#     """
#     完整流程：

#         宽度测量结果
#             -> 测量区域标签
#             -> 提取测量区域边界
#             -> 映射到原始口扫
#             -> 仅将边界标红
#     """
#     width_points = combine_width_measurement_points(
#         pcd_list_w
#     )

#     if width_points.shape[0] == 0:
#         raise ValueError("pcd_list_w 中没有有效测量点")

#     boundary_points, width_area_mask = (
#         extract_width_area_boundary(
#             reference_points=jhy_points,
#             width_points=width_points,
#             area_distance_threshold=(
#                 area_distance_threshold
#             ),
#             boundary_k=boundary_k,
#         )
#     )

#     if boundary_points.shape[0] == 0:
#         raise ValueError(
#             "未提取到测量区域边界。"
#             "可以尝试增大 area_distance_threshold。"
#         )

#     result_pcd, oral_boundary_mask = (
#         color_boundary_on_original_scan(
#             oral_scan_file=oral_scan_file,
#             boundary_points=boundary_points,
#             output_file=output_file,
#             boundary_color=(1.0, 0.0, 0.0),
#             mapping_threshold=mapping_threshold,
#             line_radius=line_radius,
#         )
#     )

#     return {
#         "pcd": result_pcd,
#         "width_points": width_points,
#         "boundary_points": boundary_points,
#         "width_area_mask": width_area_mask,
#         "oral_boundary_mask": oral_boundary_mask,
#         "output_file": output_file,
#     }

# def visualize_scalar_boundary_and_width_on_oral_scan(
#     oral_scan_file,
#     points,
#     scalar,
#     pcd_list_w,
#     output_file,
#     target_label=1,
#     boundary_k=12,
#     mapping_threshold=None,
#     boundary_line_radius=None,
#     width_line_radius=None,
# ):
#     """
#     在原始口扫上同时显示：

#         红色：scalar > 0 区域的边界
#         绿色：pcd_list_w 中的宽度测量结果
#         黄色：二者重合位置
#         其他位置：保留原始口扫颜色

#     参数
#     ----
#     oral_scan_file:
#         原始口扫文件路径。

#     points:
#         scalar 对应的完整点云坐标，通常是：
#             points = dy_obj[:, :3]

#     scalar:
#         每个点对应的标签或预测值。

#     pcd_list_w:
#         get_jhy_w() 或 oversampling() 返回的宽度测量点云列表。

#     output_file:
#         输出的彩色 PLY 文件路径。
#     """
#     points = np.asarray(
#         points,
#         dtype=np.float64,
#     )

#     scalar = np.asarray(
#         scalar
#     ).reshape(-1)

#     if points.ndim != 2 or points.shape[1] < 3:
#         raise ValueError(
#             "points 必须是 N×3 点云数组"
#         )

#     points = points[:, :3]

#     if len(points) != len(scalar):
#         raise ValueError(
#             "points 和 scalar 数量不一致："
#             f"{len(points)} != {len(scalar)}"
#         )

#     # ========================================================
#     # 1. 使用 get_surface_boundary 提取 scalar > 0 边界
#     # ========================================================
#     scalar_labels = (
#         scalar > 0
#     ).astype(np.int32)

#     if not np.any(scalar_labels > 0):
#         raise ValueError(
#             "scalar 中不存在大于 0 的点"
#         )

#     scalar_boundary_points = get_surface_boundary(
#         points=points,
#         labels=scalar_labels,
#         target_label=target_label,
#         k=boundary_k,
#     )

#     scalar_boundary_points = np.asarray(
#         scalar_boundary_points,
#         dtype=np.float64,
#     ).reshape(-1, 3)

#     if scalar_boundary_points.shape[0] == 0:
#         raise ValueError(
#             "没有提取到 scalar > 0 的边界"
#         )

#     # ========================================================
#     # 2. 直接合并所有宽度测量结果
#     #    此处不再对宽度测量结果提取边界
#     # ========================================================
#     width_point_arrays = []

#     for width_pcd in pcd_list_w:
#         if width_pcd is None:
#             continue

#         width_points = np.asarray(
#             width_pcd.points,
#             dtype=np.float64,
#         )

#         if (
#             width_points.ndim == 2
#             and width_points.shape[0] > 0
#             and width_points.shape[1] >= 3
#         ):
#             width_point_arrays.append(
#                 width_points[:, :3]
#             )

#     if not width_point_arrays:
#         raise ValueError(
#             "pcd_list_w 中没有有效的宽度测量点"
#         )

#     width_points = np.concatenate(
#         width_point_arrays,
#         axis=0,
#     )

#     # 去除 NaN 和无穷值
#     valid_width_mask = np.all(
#         np.isfinite(width_points),
#         axis=1,
#     )

#     width_points = width_points[
#         valid_width_mask
#     ]

#     if width_points.shape[0] == 0:
#         raise ValueError(
#             "宽度测量点全部无效"
#         )

#     # 去除重复点
#     _, unique_indices = np.unique(
#         np.round(width_points, decimals=6),
#         axis=0,
#         return_index=True,
#     )

#     width_points = width_points[
#         np.sort(unique_indices)
#     ]

#     # ========================================================
#     # 3. 读取原始口扫及其原始颜色
#     # ========================================================
#     oral_pcd = read_original_oral_scan(
#         oral_scan_file
#     )

#     oral_points = np.asarray(
#         oral_pcd.points,
#         dtype=np.float64,
#     )

#     if oral_points.shape[0] == 0:
#         raise ValueError(
#             f"原始口扫为空: {oral_scan_file}"
#         )

#     if oral_pcd.has_colors():
#         oral_colors = np.asarray(
#             oral_pcd.colors,
#             dtype=np.float64,
#         ).copy()
#     else:
#         oral_colors = np.tile(
#             [0.70, 0.70, 0.70],
#             (len(oral_points), 1),
#         )

#     # ========================================================
#     # 4. 自动计算映射阈值和显示线宽
#     # ========================================================
#     oral_spacing = estimate_point_spacing(
#         oral_points
#     )

#     if mapping_threshold is None:
#         mapping_threshold = max(
#             oral_spacing * 3.0,
#             0.10,
#         )

#     if boundary_line_radius is None:
#         boundary_line_radius = max(
#             oral_spacing * 1.5,
#             0.05,
#         )

#     if width_line_radius is None:
#         width_line_radius = max(
#             oral_spacing * 1.5,
#             0.05,
#         )

#     oral_tree = cKDTree(oral_points)

#     # ========================================================
#     # 5. 将任意一组点映射到原始口扫
#     # ========================================================
#     def map_points_to_oral_mask(
#         source_points,
#         line_radius,
#     ):
#         source_points = np.asarray(
#             source_points,
#             dtype=np.float64,
#         ).reshape(-1, 3)

#         oral_mask = np.zeros(
#             len(oral_points),
#             dtype=bool,
#         )

#         if source_points.shape[0] == 0:
#             return oral_mask, 0

#         distances, nearest_indices = (
#             oral_tree.query(
#                 source_points,
#                 k=1,
#             )
#         )

#         valid_mask = (
#             np.isfinite(distances)
#             & (distances <= mapping_threshold)
#         )

#         valid_points = source_points[
#             valid_mask
#         ]

#         valid_indices = nearest_indices[
#             valid_mask
#         ]

#         if valid_points.shape[0] == 0:
#             return oral_mask, 0

#         if line_radius is not None and line_radius > 0:
#             neighbor_lists = (
#                 oral_tree.query_ball_point(
#                     valid_points,
#                     r=line_radius,
#                 )
#             )

#             for neighbor_indices in neighbor_lists:
#                 if len(neighbor_indices) > 0:
#                     oral_mask[
#                         neighbor_indices
#                     ] = True
#         else:
#             oral_mask[
#                 valid_indices
#             ] = True

#         return oral_mask, len(valid_points)

#     # ========================================================
#     # 6. 映射 scalar 边界
#     # ========================================================
#     (
#         scalar_boundary_mask,
#         scalar_mapped_count,
#     ) = map_points_to_oral_mask(
#         scalar_boundary_points,
#         boundary_line_radius,
#     )

#     # ========================================================
#     # 7. 映射宽度测量结果本身
#     # ========================================================
#     (
#         width_measurement_mask,
#         width_mapped_count,
#     ) = map_points_to_oral_mask(
#         width_points,
#         width_line_radius,
#     )

#     if scalar_mapped_count == 0:
#         raise ValueError(
#             "scalar 边界无法映射到原始口扫。"
#             "请检查坐标系或增大 mapping_threshold。"
#         )

#     if width_mapped_count == 0:
#         raise ValueError(
#             "宽度测量结果无法映射到原始口扫。"
#             "请检查坐标系或增大 mapping_threshold。"
#         )

#     # ========================================================
#     # 8. 设置不同颜色
#     # ========================================================

#     # scalar > 0 的边界：红色
#     oral_colors[
#         scalar_boundary_mask
#     ] = [1.0, 0.0, 0.0]

#     # 宽度测量结果：绿色
#     oral_colors[
#         width_measurement_mask
#     ] = [0.0, 1.0, 0.0]

#     # 重合位置：黄色
#     overlap_mask = (
#         scalar_boundary_mask
#         & width_measurement_mask
#     )

#     oral_colors[
#         overlap_mask
#     ] = [1.0, 1.0, 0.0]

#     # ========================================================
#     # 9. 保存结果
#     # ========================================================
#     result_pcd = o3d.geometry.PointCloud()

#     result_pcd.points = (
#         o3d.utility.Vector3dVector(
#             oral_points
#         )
#     )

#     result_pcd.colors = (
#         o3d.utility.Vector3dVector(
#             np.clip(
#                 oral_colors,
#                 0.0,
#                 1.0,
#             )
#         )
#     )

#     output_dir = os.path.dirname(
#         output_file
#     )

#     if output_dir:
#         os.makedirs(
#             output_dir,
#             exist_ok=True,
#         )

#     success = o3d.io.write_point_cloud(
#         output_file,
#         result_pcd,
#         write_ascii=False,
#         compressed=False,
#     )

#     if not success:
#         raise IOError(
#             f"保存失败: {output_file}"
#         )

#     print(
#         "scalar 边界点数:",
#         len(scalar_boundary_points),
#     )
#     print(
#         "宽度测量点数:",
#         len(width_points),
#     )
#     print(
#         "成功映射的 scalar 边界点数:",
#         scalar_mapped_count,
#     )
#     print(
#         "成功映射的宽度测量点数:",
#         width_mapped_count,
#     )
#     print(
#         "重合着色点数:",
#         int(np.sum(overlap_mask)),
#     )
#     print(
#         "结果已保存:",
#         output_file,
#     )

#     return {
#         "pcd": result_pcd,

#         # get_surface_boundary 得到的 scalar 边界
#         "scalar_boundary_points": (
#             scalar_boundary_points
#         ),

#         # pcd_list_w 合并后的宽度测量结果
#         "width_points": width_points,

#         # 映射到原始口扫后的掩码
#         "scalar_boundary_mask": (
#             scalar_boundary_mask
#         ),
#         "width_measurement_mask": (
#             width_measurement_mask
#         ),
#         "overlap_mask": overlap_mask,

#         "output_file": output_file,
#     }

def visualize_scalar_boundary_and_width_on_oral_scan(
    oral_scan_file,
    points,
    scalar,
    pcd_list_w,
    output_file,
    target_label=1,
    boundary_k=12,
    width_area_threshold=None,
    mapping_threshold=None,
    boundary_line_radius=None,
):
    """
    在原始口扫上只显示两种边界：

        红色：scalar > 0 区域的边界
        绿色：宽度测量结果所形成区域的边界
        黄色：两种边界的重合部分
        其他：保留原始口扫颜色
    """
    points = np.asarray(points, dtype=np.float64)
    scalar = np.asarray(scalar).reshape(-1)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points 必须为 N×3 数组")

    points = points[:, :3]

    if len(points) != len(scalar):
        raise ValueError(
            f"points 和 scalar 数量不一致："
            f"{len(points)} != {len(scalar)}"
        )

    # ========================================================
    # 1. 提取 scalar > 0 的边界
    # ========================================================
    scalar_labels = (scalar > 0).astype(np.int32)

    if not np.any(scalar_labels == target_label):
        raise ValueError("scalar 中不存在大于 0 的点")

    scalar_boundary_points = get_surface_boundary(
        points=points,
        labels=scalar_labels,
        target_label=target_label,
        k=boundary_k,
    )

    scalar_boundary_points = np.asarray(
        scalar_boundary_points,
        dtype=np.float64,
    ).reshape(-1, 3)

    if scalar_boundary_points.shape[0] == 0:
        raise ValueError("没有提取到 scalar 边界")

    # ========================================================
    # 2. 合并宽度测量结果
    # ========================================================
    width_arrays = []

    for width_pcd in pcd_list_w:
        if width_pcd is None:
            continue

        current_points = np.asarray(
            width_pcd.points,
            dtype=np.float64,
        )

        if (
            current_points.ndim == 2
            and current_points.shape[0] > 0
            and current_points.shape[1] >= 3
        ):
            width_arrays.append(current_points[:, :3])

    if not width_arrays:
        raise ValueError(
            "pcd_list_w 中没有有效宽度测量点"
        )

    width_points = np.concatenate(
        width_arrays,
        axis=0,
    )

    width_points = width_points[
        np.all(np.isfinite(width_points), axis=1)
    ]

    if width_points.shape[0] == 0:
        raise ValueError("宽度测量点全部无效")

    _, unique_indices = np.unique(
        np.round(width_points, decimals=6),
        axis=0,
        return_index=True,
    )

    width_points = width_points[
        np.sort(unique_indices)
    ]

    # ========================================================
    # 3. 把宽度测量结果转换成 points 上的区域标签
    #
    # 不能直接对 pcd_list_w 调用 get_surface_boundary：
    # pcd_list_w 通常只包含区域内部点，没有区域外点。
    # 必须先映射到完整点云 points，形成 0/1 标签。
    # ========================================================
    width_tree = cKDTree(width_points)

    width_distances, _ = width_tree.query(
        points,
        k=1,
    )

    if width_area_threshold is None:
        point_tree = cKDTree(points)

        neighbor_distances, _ = point_tree.query(
            points,
            k=min(2, len(points)),
        )

        if (
            neighbor_distances.ndim == 2
            and neighbor_distances.shape[1] > 1
        ):
            spacing_values = neighbor_distances[:, 1]
            spacing_values = spacing_values[
                np.isfinite(spacing_values)
                & (spacing_values > 1e-8)
            ]

            if spacing_values.shape[0] > 0:
                point_spacing = float(
                    np.median(spacing_values)
                )
            else:
                point_spacing = 0.05
        else:
            point_spacing = 0.05

        width_area_threshold = max(
            point_spacing * 2.5,
            0.08,
        )

    width_labels = (
        width_distances <= width_area_threshold
    ).astype(np.int32)

    if not np.any(width_labels == target_label):
        raise ValueError(
            "宽度测量结果没有映射到完整点云。"
            "请增大 width_area_threshold。"
        )

    # ========================================================
    # 4. 提取宽度测量区域边界
    # ========================================================
    width_boundary_points = get_surface_boundary(
        points=points,
        labels=width_labels,
        target_label=target_label,
        k=boundary_k,
    )

    width_boundary_points = np.asarray(
        width_boundary_points,
        dtype=np.float64,
    ).reshape(-1, 3)

    if width_boundary_points.shape[0] == 0:
        raise ValueError(
            "没有提取到宽度测量结果边界。"
            "请调整 width_area_threshold。"
        )

    # ========================================================
    # 5. 读取原始口扫
    # ========================================================
    oral_pcd = read_original_oral_scan(
        oral_scan_file
    )

    oral_points = np.asarray(
        oral_pcd.points,
        dtype=np.float64,
    )

    if oral_points.shape[0] == 0:
        raise ValueError(
            f"原始口扫为空: {oral_scan_file}"
        )

    if oral_pcd.has_colors():
        oral_colors = np.asarray(
            oral_pcd.colors,
            dtype=np.float64,
        ).copy()
    else:
        oral_colors = np.tile(
            [0.70, 0.70, 0.70],
            (len(oral_points), 1),
        )

    oral_tree = cKDTree(oral_points)

    # 估计原始口扫点间距
    oral_neighbor_distances, _ = oral_tree.query(
        oral_points,
        k=min(2, len(oral_points)),
    )

    if (
        oral_neighbor_distances.ndim == 2
        and oral_neighbor_distances.shape[1] > 1
    ):
        oral_spacing_values = (
            oral_neighbor_distances[:, 1]
        )

        oral_spacing_values = oral_spacing_values[
            np.isfinite(oral_spacing_values)
            & (oral_spacing_values > 1e-8)
        ]

        if oral_spacing_values.shape[0] > 0:
            oral_spacing = float(
                np.median(oral_spacing_values)
            )
        else:
            oral_spacing = 0.05
    else:
        oral_spacing = 0.05

    if mapping_threshold is None:
        mapping_threshold = max(
            oral_spacing * 3.0,
            0.10,
        )

    if boundary_line_radius is None:
        boundary_line_radius = max(
            oral_spacing * 1.5,
            0.05,
        )

    # ========================================================
    # 6. 将边界映射到原始口扫
    # ========================================================
    def map_boundary_to_oral(boundary_points):
        oral_mask = np.zeros(
            len(oral_points),
            dtype=bool,
        )

        distances, nearest_indices = oral_tree.query(
            boundary_points,
            k=1,
        )

        valid_mask = (
            np.isfinite(distances)
            & (distances <= mapping_threshold)
        )

        valid_boundary_points = boundary_points[
            valid_mask
        ]

        valid_nearest_indices = nearest_indices[
            valid_mask
        ]

        if valid_boundary_points.shape[0] == 0:
            return oral_mask, 0

        if (
            boundary_line_radius is not None
            and boundary_line_radius > 0
        ):
            neighbor_lists = (
                oral_tree.query_ball_point(
                    valid_boundary_points,
                    r=boundary_line_radius,
                )
            )

            for indices in neighbor_lists:
                if len(indices) > 0:
                    oral_mask[indices] = True
        else:
            oral_mask[
                valid_nearest_indices
            ] = True

        return oral_mask, len(valid_boundary_points)

    (
        scalar_boundary_mask,
        scalar_mapped_count,
    ) = map_boundary_to_oral(
        scalar_boundary_points
    )

    (
        width_boundary_mask,
        width_mapped_count,
    ) = map_boundary_to_oral(
        width_boundary_points
    )

    if scalar_mapped_count == 0:
        raise ValueError(
            "scalar 边界无法映射到原始口扫"
        )

    if width_mapped_count == 0:
        raise ValueError(
            "宽度边界无法映射到原始口扫"
        )

    # ========================================================
    # 7. 只给两种边界着色
    # ========================================================

    # scalar 边界：红色
    oral_colors[
        scalar_boundary_mask
    ] = [1.0, 0.0, 0.0]

    # 宽度测量区域边界：绿色
    oral_colors[
        width_boundary_mask
    ] = [0.0, 1.0, 0.0]

    # 两条边界重合：黄色
    overlap_mask = (
        scalar_boundary_mask
        & width_boundary_mask
    )

    oral_colors[
        overlap_mask
    ] = [1.0, 1.0, 0.0]

    # ========================================================
    # 8. 保存结果
    # ========================================================
    result_pcd = o3d.geometry.PointCloud()

    result_pcd.points = o3d.utility.Vector3dVector(
        oral_points
    )

    result_pcd.colors = o3d.utility.Vector3dVector(
        np.clip(oral_colors, 0.0, 1.0)
    )

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # success = o3d.io.write_point_cloud(
    #     output_file,
    #     result_pcd,
    #     write_ascii=False,
    #     compressed=False,
    # )

    # if not success:
    #     raise IOError(f"保存失败: {output_file}")

    # print("scalar 边界点数:", len(scalar_boundary_points))
    # print("宽度边界点数:", len(width_boundary_points))
    # print("映射后的 scalar 边界点数:", scalar_mapped_count)
    # print("映射后的宽度边界点数:", width_mapped_count)
    # print("两种边界重合点数:", int(np.sum(overlap_mask)))
    # print("结果已保存:", output_file)

    return {
        "pcd": result_pcd,
        "scalar_boundary_points": scalar_boundary_points,
        "width_boundary_points": width_boundary_points,
        "scalar_boundary_mask": scalar_boundary_mask,
        "width_boundary_mask": width_boundary_mask,
        "overlap_mask": overlap_mask,
        "width_labels": width_labels,
        "output_file": output_file,
    }


def visualize_buccal_2mm_points(points, axiox2):
    points = np.asarray(points, dtype=float)
    axiox2 = normalize(axiox2)

    selected_points, mask = get_buccal_2mm_points(points, axiox2)

    projection = np.dot(points, axiox2)
    buccal_boundary = np.min(projection)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 全部点：灰色
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="lightgray",
        s=5,
        label="全部点"
    )

    # 返回的点：红色
    ax.scatter(
        selected_points[:, 0],
        selected_points[:, 1],
        selected_points[:, 2],
        c="red",
        s=15,
        label="颊侧 2 mm 内的点"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"Buccal points: {len(selected_points)} / {len(points)}"
    )
    ax.legend()

    plt.tight_layout()
    plt.show()

    return selected_points, mask


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return np.zeros_like(vector)

    return vector / norm


def find_oral_scan_file(oral_scan_dir, base_name):
    for ext in [".txt", ".ply"]:
        file_path = os.path.join(
            oral_scan_dir,
            base_name + ext,
        )

        if os.path.exists(file_path):
            return file_path

    return None


def read_oral_scan(file_path):
    if file_path.lower().endswith(".ply"):
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)

    data = np.loadtxt(file_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data[:, :3]


def get_buccal_2mm_points(points, axiox2):
    """
    axiox2 约定为：颊侧 -> 舌侧。

    投影最小的位置是颊侧边界，
    从颊侧边界向舌侧保留 2 mm。
    """
    points = np.asarray(points, dtype=float)
    axiox2 = normalize(axiox2)

    projection = np.dot(points, axiox2)

    buccal_boundary = np.min(projection)

    mask = projection <= (
        buccal_boundary + BUCCAL_KEEP_MM
    )

    return points[mask], mask


def save_visualization(
    points,
    mask,
    output_file,
):
    """
    红色：颊侧 2 mm
    灰色：其他角化龈点
    """
    points = np.asarray(points)

    colors = np.tile(
        [0.65, 0.65, 0.65],
        (points.shape[0], 1),
    )

    colors[mask] = [1.0, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    # visual_dir = os.path.join(
    #     args.output,
    #     "visualization",
    # )

    # os.makedirs(visual_dir, exist_ok=True)

    output_name = os.path.basename(
        os.path.normpath(args.output)
    ).lower()

    if output_name == "gt":
        prefix = "gt"
        scalar_index = 6
    else:
        prefix = "ai"
        scalar_index = 7

    files = sorted(
        os.listdir(args.keratinized_gingiva)
    )

    for file in files:
        dy_file = os.path.join(
            args.keratinized_gingiva,
            file,
        )

        if not os.path.isfile(dy_file):
            continue

        print("\n处理:", file)

        # 与原代码保持一致
        if len(file) > 8:
            base_name = file[:-8]
        else:
            base_name = os.path.splitext(file)[0]

        oral_scan_file = find_oral_scan_file(
            args.rotake_oral_scan,
            base_name,
        )

        if oral_scan_file is None:
            print("找不到对应口扫文件:", base_name)
            continue

        try:
            dy_obj = np.loadtxt(dy_file)
        except Exception as exc:
            print("读取失败:", dy_file, exc)
            continue

        if dy_obj.ndim == 1:
            dy_obj = dy_obj.reshape(1, -1)

        if dy_obj.shape[1] <= scalar_index:
            print("列数不足，无法读取 scalar:", file)
            continue

        points = dy_obj[:, :3].astype(float)
        colors = dy_obj[:, 3:6].astype(float)
        scalar = dy_obj[:, scalar_index].astype(float)

        if np.max(colors) > 1.0:
            colors = colors / 255.0

        # 建立原始点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd1 = copy.deepcopy(pcd)
        pcd2 = copy.deepcopy(pcd)

        # HSV 过滤和聚类
        try:
            filtered_pcd, labels = filt_rpoin_hsv(pcd)
            filtered_pcd, labels = use_dbscan(filtered_pcd)
        except Exception as exc:
            print("点云过滤或聚类失败:", exc)
            continue

        filtered_points = np.asarray(
            filtered_pcd.points
        )

        centers = []

        for label in np.unique(labels):
            cluster = filtered_points[labels == label]

            if cluster.shape[0] > 0:
                centers.append(
                    np.mean(cluster, axis=0)
                )

        centers = np.asarray(centers)

        if centers.shape[0] < 2:
            print("聚类中心少于 2 个:", file)
            continue

        # 读取口扫并计算中心
        try:
            oral_points = read_oral_scan(
                oral_scan_file
            )
        except Exception as exc:
            print("读取口扫失败:", exc)
            continue

        if oral_points.shape[0] == 0:
            print("口扫点云为空:", file)
            continue

        oral_center = np.mean(
            oral_points,
            axis=0,
        )

        # 以距离口扫中心排序
        distances = np.linalg.norm(
            centers - oral_center,
            axis=1,
        )

        centers = centers[np.argsort(distances)]

        # 近中-远中方向
        axiox1 = normalize(
            centers[1] - centers[0]
        )

        # 根据原代码计算颊舌方向
        center_x_array = np.column_stack(
            (
                -centers[:, 1],
                centers[:, 0],
                np.zeros(centers.shape[0]),
            )
        )

        candidate0 = center_x_array[0]
        candidate1 = center_x_array[1]

        dist0 = np.linalg.norm(
            candidate0 - oral_center
        )

        dist1 = np.linalg.norm(
            candidate1 - oral_center
        )

        outer_point = (
            candidate0
            if dist0 >= dist1
            else candidate1
        )

        toward_oral = normalize(
            oral_center - outer_point
        )

        toward_oral_proj = (
            toward_oral
            - np.dot(toward_oral, axiox1) * axiox1
        )


        if np.linalg.norm(toward_oral_proj) < 1e-8:
            axiox2 = np.array(
                [0.0, 0.0, 1.0]
            )
        else:
            axiox2 = normalize(
                toward_oral_proj
            )
        axiox2 = normalize(toward_oral_proj)

        print('axiox1:',axiox1,'axiox2:',axiox2)
        print("垂直性 =", np.dot(axiox1, axiox2))
        # axiox2 约定为颊侧 -> 舌侧
        #
        # 如果可视化发现红色在舌侧，
        # 将下面这一行取消注释：
        #
        if(np.dot(axiox1, axiox2)>0):
            axiox2 = axiox2
        else:
            axiox2 = -axiox2

        axiox3 = normalize(
            np.cross(axiox1, axiox2)
        )

        center_point = (
            centers[0] + centers[1]
        ) / 2.0

        x_line, y_line, z_line = create_coordinate_frame(
            center_point,
            axiox1,
            axiox2,
            axiox3,
        )

        centers_pcd = get_poin_list(
            centers,
            [[1, 0, 0]],
        )

        # 只保留角化龈点
        valid_mask = scalar > 0

        jhy_points = points[valid_mask]
        jhy_colors = colors[valid_mask]

        if jhy_points.shape[0] < 2:
            print("角化龈有效点少于 2 个:", file)
            continue


        cent_list_for_measure = [
            axiox1,
            axiox2,
            axiox3,
            pcd1,
            centers_pcd,
        ]

        vis_list = [
            x_line,
            y_line,
            z_line,
        ]

        axiox_list = [
            axiox1,
            axiox2,
            axiox3,
        ]

        # 只测量颊侧 2 mm 宽度
        try:
            pcd_list_w, dist_w, offsets_w = get_jhy_w(
                # buccal_2mm_points,
                # buccal_colors,
                jhy_points,
                jhy_colors,
                cent_list_for_measure,
                SLICE_STEP_MM,
                vis_list,
                axiox_list,
                oral_center.reshape(1, 3),
                return_offsets=True,
            )
        except Exception as exc:
            print("颊侧宽度测量失败:", exc)
            continue

        # print("颊侧 2 mm 宽度:", dist_w)
        pcd_list_w, dist_w, offsets_w = oversampling(
            pcd_list_w,
            dist_w,
            offsets_w,
            oversample_num=5,
            sample_points=200,
        )


        # ============================================================
        # 将宽度测量区域的边界叠加到原始口扫
        # ============================================================
        boundary_output_dir = os.path.join(
            args.output,
            "width_boundary_on_oral_scan",
        )
        
        os.makedirs(
            boundary_output_dir,
            exist_ok=True,
        )
        
        boundary_output_file = os.path.join(
            boundary_output_dir,
            prefix
            + "_"
            + base_name
            + "_width_boundary.ply",
        )
        
        # try:
        #     boundary_result = (
        #         visualize_width_boundary_on_oral_scan(
        #             oral_scan_file=oral_scan_file,
        
        #             # 第二段代码中提取的角化龈点
        #             jhy_points=jhy_points,
        
        #             # 宽度测量及过采样后的结果
        #             pcd_list_w=pcd_list_w,
        
        #             # 输出：原始口扫 + 红色边界
        #             output_file=boundary_output_file,
        
        #             # None 表示自动估算
        #             area_distance_threshold=None,
        #             mapping_threshold=None,
        
        #             # 与第一段边界提取参数一致
        #             boundary_k=12,
        
        #             # None 表示自动设置边界线宽
        #             line_radius=None,
        #         )
        #     )
        
        # except Exception as exc:
        #     print(
        #         "宽度边界叠加到原始口扫失败:",
        #         file,
        #         exc,
        #     )

        try:
            # result = (
            #     visualize_scalar_boundary_and_width_on_oral_scan(
            #         oral_scan_file=oral_scan_file,
        
            #         # dy_obj 的完整坐标
            #         points=points,
        
            #         # AI 或 GT 的 scalar
            #         scalar=scalar,
        
            #         # 宽度测量并过采样后的结果
            #         pcd_list_w=pcd_list_w,
        
            #         output_file=boundary_output_file,
        
            #         target_label=1,
            #         boundary_k=12,
        
            #         # None 表示自动计算
            #         mapping_threshold=None,
            #         boundary_line_radius=None,
            #         width_line_radius=None,
            #     )
            # )
            result = (
                visualize_scalar_boundary_and_width_on_oral_scan(
                    oral_scan_file=oral_scan_file,
                    points=points,
                    scalar=scalar,
                    pcd_list_w=pcd_list_w,
                    output_file=boundary_output_file,
            
                    target_label=1,
                    boundary_k=12,
            
                    # 宽度结果映射成区域标签的距离阈值
                    width_area_threshold=None,
            
                    # 两种边界映射到原始口扫的距离阈值
                    mapping_threshold=None,
            
                    # 两条边界在原始口扫上的显示粗细
                    boundary_line_radius=None,
                )
            )
        
        except Exception as exc:
            print(
                "scalar 边界和宽度测量结果可视化失败:",
                file,
                exc,
            )



        converter = PointCloudMeshConverter(
            radius_factors=(3.5, 4.5, 6.0),
            min_component_triangles=0,
            save_normals=False
        )
        mesh = converter.build_mesh(result['pcd'])
        o3d.io.write_triangle_mesh(
            os.path.join(boundary_output_dir, prefix + "_" + base_name + "_width_boundary_mesh.ply"),
            mesh,
            write_ascii=False,
            write_vertex_normals=False,
            write_vertex_colors=True
        )


        # if len(pcd_list_w) == 0:
        #     print("pcd_list_w 为空，跳过当前文件:", file)
        #     continue
        # else:
        #     # 保存切片点云
        #     save_to_txt(
        #         pcd2,
        #         pcd_list_w,
        #         args.output,
        #         prefix + "_buccal_2mm_" + file,
        #         scalar,
        #         "buccal_2mm_wid",
        #         offsets_w,
        #         # TARGET_OFFSETS,
        #         offsets_w
        #     )

        print("完成:", file)


if __name__ == "__main__":
    main()