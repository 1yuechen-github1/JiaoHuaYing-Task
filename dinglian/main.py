import csv
import os
import numpy as np
import open3d as o3d
from path import Path
from utils import *
import copy
import argparse

# path = r"C:\yuechen\code\jiaohuaying\1.code\0110\data\1106"
# output = r"C:\yuechen\code\jiaohuaying\1.code\0110\data\output2\医生"

def get_args_parser():
    parser = argparse.ArgumentParser(description='此代码用于输出角化龈的定量分析')
    parser.add_argument("--input",dest='input', type=str,help="点云输入路径")
    parser.add_argument("--output", dest='output',type=str,help="结果输出路径")
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output:
        os.makedirs(args.output,exist_ok=True)
    for file in os.listdir(args.input):
        path = args.input
        print(file)
        dy_file = path + '\\' + file
        dy_obj = np.loadtxt(dy_file)
        points = dy_obj[:,:3]
        colors = dy_obj[:,3:6]
        colors = colors / 255.0
        # 网络
        scalar = dy_obj[:, -1]
        # 医生
        # scalar = dy_obj[:,6:7]  # (N,)
        scalar = scalar.astype(float)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd1 = pcd
        pcd2 = copy.deepcopy(pcd)
        # pcd1.colors = o3d.utility.Vector3dVector(colors1)
        pcd, red_pcd = filt_rpoin_hsv(pcd)
        pcd,labels = use_dbscan(pcd)
        # print('labels',len(labels))
        # vis([pcd2], f"{file} - 三轴坐标系")
        points1 = points
        colors1 = colors
        for i in range(len(points1)):
            if scalar[i] > 0:
                colors1[i] = [0, 0, 1]
            elif scalar[i] == 0:
                colors1[i] = colors1[i]

        pcd1.colors = o3d.utility.Vector3dVector(colors1)
        cent_list = []
        center_x_list = []
        for label in np.unique(labels):
            mask = (label == labels)
            mk_poin = np.asarray(pcd.points)[mask]
            center = np.mean(mk_poin, axis = 0)
            center_x = [-center[1],center[0],0]
            cent_list.append(center)
            center_x_list.append(center_x)
        centers_array = np.array(cent_list)
        centers_pcd = get_poin_list(centers_array, [[1, 0, 0]])
        centers_x_array = np.array(center_x_list)
        centers_x_pcd = get_poin_list(centers_x_array, [[0, 0, 1]])
        axiox1 = centers_array[1] - centers_array[0]
        axiox1 = axiox1 / np.linalg.norm(axiox1)
        axiox2 = centers_x_array[1] - centers_x_array[0]
        axiox2 = axiox2 / np.linalg.norm(axiox2)
        axiox3 = np.cross(axiox1, axiox2)
        axiox3 = axiox3 / np.linalg.norm(axiox3)
        center_point = (np.array(cent_list[0]) + np.array(cent_list[1])) / 2
        x_line, y_line, z_line = create_coordinate_frame(center_point, axiox1, axiox2, axiox3)
        centers_array = np.array(cent_list)
        centers_pcd = get_poin_list(centers_array, [[1, 0, 0]])  # 红色
        centers_x_array = np.array(center_x_list)
        centers_x_pcd = get_poin_list(centers_x_array, [[0, 0, 1]])  # 蓝色

        red_indices = np.where(scalar > 0)[0]
        points_array = np.asarray(pcd1.points)
        jhy_poins = points_array[red_indices]
        jhy_center = np.mean(jhy_poins, axis=0)

        print('x_line, y_line, z_line',axiox1, axiox2, axiox3, jhy_poins.shape)
        cent_list = [axiox1, axiox2, axiox3,pcd1, centers_pcd]
        # 高度/宽度测量
        # pcd_list_h, dist_h= get_jhy_h(jhy_poins,jhy_center,cent_list,  1, file)
        # pcd_list_w, dist_w = get_jhy_w(jhy_poins, jhy_center, cent_list, 1, file)
        vis_list_h = [x_line, y_line, z_line]
        axiox_list = [axiox1, axiox2, axiox3]
        pcd_list_h, dist_h= get_jhy_h(jhy_poins,cent_list, 1,vis_list_h,axiox_list)
        pcd_list_w, dist_w = get_jhy_w(jhy_poins, cent_list, 1,vis_list_h,axiox_list)
        output = args.output
        save_to_txt(pcd2, pcd_list_h, output, file, scalar, 'hig')
        save_to_txt(pcd2, pcd_list_w, output, file, scalar, 'wid')

        # vis_list_h = [pcd2, centers_pcd,x_line, y_line, z_line]
        # pcd_list_h.extend(vis_list_h)
        # vis(pcd_list_h, f"{file} - 三轴坐标系")
        #
        # vis_list_w = [pcd2, centers_pcd,x_line, y_line, z_line]
        # pcd_list_w.extend(vis_list_w)
        # vis(pcd_list_w, f"{file} - 三轴坐标系")

        index_list = [0, -1, 1, -2, 2, -3, 3]

        with open(output + '\\jhy_h.csv', 'a', encoding='utf-8-sig') as f:
            for idx, dist in zip(index_list, dist_h):
                f.write(f"{file}角化龈距离中心{idx}mm,{dist}\n")

        with open(output + '\\jhy_w.csv', 'a', encoding='utf-8-sig') as f:
            for idx, dist in zip(index_list, dist_w):
                f.write(f"{file}角化龈距离中心{idx}mm,{dist}\n")

