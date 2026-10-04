import argparse
import copy
import os

import numpy as np
import open3d as o3d

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


# python code\doc\KeratinizedGumGraft\main.py 
# --keratinized_gingiva Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\pointcept-加上下颌位置信息-法向量-边界做数据增强\result-txt-test 
# --rotake_oral_scan Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\test\rotake_oral_scan 
# --output Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\pointcept-加上下颌位置信息-法向量-边界做数据增强\output1

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


import matplotlib.pyplot as plt
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

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True,
    )

    o3d.io.write_point_cloud(
        output_file,
        pcd,
    )


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    visual_dir = os.path.join(
        args.output,
        "visualization",
    )

    os.makedirs(visual_dir, exist_ok=True)

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

        # 获取颊侧 2 mm 点
        buccal_2mm_points, buccal_mask = (
            get_buccal_2mm_points(
                jhy_points,
                axiox2,
            )
        )

        # selected_points, mask = visualize_buccal_2mm_points(
        #     buccal_2mm_points,
        #     axiox2
        # )

        # if buccal_2mm_points.shape[0] < 2:
        #     print("颊侧 2 mm 点少于 2 个:", file)
        #     continue

        # 保存可视化
        visual_file = os.path.join(
            visual_dir,
            prefix
            + "_"
            + os.path.splitext(file)[0]
            + "_buccal_2mm.ply",
        )

        save_visualization(
            jhy_points,
            buccal_mask,
            visual_file,
        )

        # print("可视化:", visual_file)

        # 颊侧 2 mm 点颜色
        # buccal_colors = np.tile(
        #     [1.0, 0.0, 0.0],
        #     (buccal_2mm_points.shape[0], 1),
        # )

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

        if len(pcd_list_w) == 0:
            print("pcd_list_w 为空，跳过当前文件:", file)
            continue
        else:
            # 保存切片点云
            save_to_txt(
                pcd2,
                pcd_list_w,
                args.output,
                prefix + "_buccal_2mm_" + file,
                scalar,
                "buccal_2mm_wid",
                offsets_w,
                # TARGET_OFFSETS,
                offsets_w
            )

        print("完成:", file)


if __name__ == "__main__":
    main()