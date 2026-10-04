import argparse
import csv
import copy
import os

import numpy as np
import open3d as o3d

from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser(description="用于角化龈点云的定量分析")
    parser.add_argument("--keratinized_gingiva", type=str, help="角化龈点云目录")
    parser.add_argument("--rotake_oral_scan", type=str, help="口扫点云目录")
    parser.add_argument("--output", type=str, help="输出目录")
    return parser



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    for file in os.listdir(args.keratinized_gingiva):
        print('file:',file)
        dy_file = os.path.join(args.keratinized_gingiva, file)
        if not os.path.isfile(dy_file):
            continue
        base_name = file[0:-8]
        is_upper = base_name.split("_")[-1]
        oral_scan_file = None
        for ext in (".txt", ".ply"):
            candidate_file = os.path.join(args.rotake_oral_scan, base_name + ext)
            if os.path.exists(candidate_file):
                oral_scan_file = candidate_file
                break
        if oral_scan_file is None:
            print("skip, oral scan missing:", os.path.join(args.rotake_oral_scan, base_name + ".txt/.ply"))
            continue

        # 读取角化龈点云
        dy_obj = np.loadtxt(dy_file)
        points = dy_obj[:, :3]
        colors = dy_obj[:, 3:6] / 255.0
        prefix = args.output.split("\\")[-1]
        if prefix == 'gt':
            scalar = dy_obj[:, 6:7].astype(float)  #这是第7列
        else:
            prefix = 'ai'
            scalar = dy_obj[:, 7:8].astype(float)  #这是第8列
         
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd1 = pcd
        pcd2 = copy.deepcopy(pcd)

        pcd, _ = filt_rpoin_hsv(pcd) #使用 HSV 颜色空间过滤，保留非红色点。 
        pcd, labels = use_dbscan(pcd)

        # 读取口扫点云并计算中心点
        if oral_scan_file.lower().endswith(".ply"):
            oral_scan_obj = o3d.io.read_point_cloud(oral_scan_file)
            pcd_oral_scan_points = np.asarray(oral_scan_obj.points)
        else:
            oral_scan_obj = np.loadtxt(oral_scan_file)
            pcd_oral_scan_points = oral_scan_obj[:, :3]
        oral_scan_center = np.mean(pcd_oral_scan_points, axis=0).reshape(1, 3)
        oral_scan_center_point = oral_scan_center[0]

        # 计算聚类中心
        cent_list = []
        for label in np.unique(labels):
            mask = label == labels
            mk_poin = np.asarray(pcd.points)[mask]
            center = np.mean(mk_poin, axis=0)
            cent_list.append(center)

        centers_array = np.array(cent_list)
        if centers_array.shape[0] < 2:
            print("skip, cluster centers < 2:", file)
            continue

        # 按到口扫中心的距离排序：近中、远中
        dists = np.linalg.norm(centers_array - oral_scan_center_point, axis=1)
        sort_idx = np.argsort(dists)
        centers_array = centers_array[sort_idx]

        # 近中-远中轴
        axiox1 = centers_array[1] - centers_array[0]
        axiox1 = axiox1 / np.linalg.norm(axiox1)

        # 构造候选点，并判断哪个在点云外部
        center_x_array = np.column_stack(
            (-centers_array[:, 1], centers_array[:, 0], np.zeros(centers_array.shape[0]))
        )
        candidate0 = center_x_array[0]
        candidate1 = center_x_array[1]
        dist0 = np.linalg.norm(candidate0 - oral_scan_center_point)
        dist1 = np.linalg.norm(candidate1 - oral_scan_center_point)
        outer_point = candidate0 if dist0 >= dist1 else candidate1

        # 舌腭方向：oral_scan_center - 点云外部点
        toward_oral = oral_scan_center_point - outer_point
        toward_oral = toward_oral / np.linalg.norm(toward_oral)

        # 将方向投影到与 axiox1 垂直的平面，得到第二轴
        toward_oral_proj = toward_oral - np.dot(toward_oral, axiox1) * axiox1
        if np.linalg.norm(toward_oral_proj) < 1e-8:
            axiox2 = np.array([0.0, 0.0, 1.0])
        else:
            axiox2 = toward_oral_proj / np.linalg.norm(toward_oral_proj)

        # 第三轴
        axiox3 = np.cross(axiox1, axiox2)
        axiox3 = axiox3 / np.linalg.norm(axiox3)

        center_point = (centers_array[0] + centers_array[1]) / 2.0
        x_line, y_line, z_line = create_coordinate_frame(center_point, axiox1, axiox2, axiox3)
        centers_pcd = get_poin_list(centers_array, [[1, 0, 0]])

        # 仅取角化龈区域点
        red_indices = np.where(scalar > 0)[0]
        points_array = np.asarray(pcd1.points)
        colors_array = np.asarray(pcd1.colors)
        jhy_points = points_array[red_indices]
        jhy_colors = colors_array[red_indices]
        if jhy_points.shape[0] == 0:
            print("skip, no scalar>0 points:", file)
            continue

        # 测量：中央按 z 最大切片，切片范围自动扩展
        cent_list_for_measure = [axiox1, axiox2, axiox3, pcd1, centers_pcd]
        vis_list_h = [x_line, y_line, z_line]
        axiox_list = [axiox1, axiox2, axiox3]

        pcd_list_h, dist_h, offsets_h = get_jhy_h(
            jhy_points,
            jhy_colors,
            cent_list_for_measure,
            0.5,
            vis_list_h,
            axiox_list,
            oral_scan_center,
            is_upper,
            return_offsets=True,
        )
        print('dist_h:',dist_h)
        print('offsets_h:',offsets_h)
        pcd_list_w, dist_w, offsets_w = get_jhy_w(
            jhy_points,
            jhy_colors,
            cent_list_for_measure,
            0.5,
            vis_list_h,
            axiox_list,
            oral_scan_center,
            return_offsets=True,
        )

        output = args.output
        is_upper_case = str(is_upper).lower()
        height_csv_offsets = (
            [off for off in TARGET_OFFSETS if off >= 0]
            if is_upper_case in {"upper", "up", "u", "maxilla"}
            else TARGET_OFFSETS
        )
        save_to_txt(
            pcd2, pcd_list_h, output, prefix+'_'+file, scalar, "hig",
            offsets_h, height_csv_offsets,
        )
        save_to_txt(
            pcd2, pcd_list_w, output, prefix+'_'+file, scalar, "wid",
            offsets_w, TARGET_OFFSETS,
        )


        # CSV：每个文件一行，固定输出 0, -1, 1, -2, 2, -3, 3 七个切片位置
        print("file:",file,offsets_h,dist_h,label_h,is_upper)
        # if is_upper == 'upper':
        if 'upp' in file or 'Upp' in file:
            append_measure_csv(
                os.path.join(output, f"{prefix}_jhy_upper_h.csv"),
                file,
                offsets_h,
                dist_h,
                label_h,
                is_upper
            )
        else:
            append_measure_csv(
                os.path.join(output, f"{prefix}_jhy_lower_h.csv"),
                file,
                offsets_h,
                dist_h,
                label_h,
                is_upper
            )            
        append_measure_csv(
            os.path.join(output, f"{prefix}_jhy_w.csv"),
            file,
            offsets_w,
            dist_w,
            label_w,
            None
        )

