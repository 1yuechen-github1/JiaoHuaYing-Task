import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import cv2
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from sklearn.cluster import DBSCAN
from glob import glob
import re
# import torch
from utils import *
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 主函数
def main():
    # 文件夹路径
    input_folder = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\1.AllData-PointCloud'
    output_folder = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\1.AllData-MissingToothArea\cropped_debug'
    cropped_output_folder = os.path.join(output_folder, 'cropped_debug')
    
    # nnU-Net模型路径
    # nnunet_model_folder = r'/home/yongkang/nnunet/nnUnet_local/data/dataset1/nnUNet_trained_models/Dataset129_jiaohuaying175/nnUNetTrainer__nnUNetPlans__2d/'


    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(cropped_output_folder, exist_ok=True)
    logging.info(f"确保输出文件夹存在: {output_folder}")

    # 初始化nnU-Net预测器（使用标准参数）
    logging.info("初始化nnU-Net预测器")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # predictor = nnUNetPredictor(
    #     tile_step_size=0.5,
    #     use_gaussian=True,
    #     use_mirroring=True,
    #     perform_everything_on_device=True,
    #     device=device,
    #     verbose=True,
    #     allow_tqdm=True
    # )
    
    # 初始化模型
    # predictor.initialize_from_trained_model_folder(nnunet_model_folder, use_folds=[0])

    # 加载所有点云文件
    point_clouds, original_points_list, original_colors_list, filenames = load_all_point_clouds_from_folder(input_folder)

    # 存储生成的图像路径和图像信息
    image_paths = []
    img_info_list = []
    rotated_point_cloud_list = []
    rotated_point_cloud_color_list = []

    # 处理每个点云
    projection_2d_save_path = os.path.join(output_folder,'2d_projection')
    # cropped_output_folder = os.path.join()
    os.makedirs(projection_2d_save_path,exist_ok=True)
    for i, (point_cloud, filename) in enumerate(zip(point_clouds, filenames)):
        logging.info(f"开始处理文件 {i+1}/{len(filenames)}: {filename}")

        # jaw_type = classify_jaw_type(os.path.join(input_folder, filename))
        # logging.info(f"文件 {filename} 被分类为: {jaw_type}")
        if 'upper' in filename or 'Upper' in filename or 'Upp' in filename or 'upp' in filename:
            rotated_point_cloud = rotate_point_cloud(point_cloud, axis='y', angle_deg=180)
            rotated_point_cloud = rotate_point_cloud(rotated_point_cloud, axis='x', angle_deg=-35)
            rotated_point_cloud = rotate_point_cloud(rotated_point_cloud, axis='z', angle_deg=180)

        else:
            rotated_point_cloud = rotate_point_cloud(point_cloud, axis='x', angle_deg=-20)

        xyz_coordinates = np.asarray(rotated_point_cloud.points)
        rgb_colors = np.asarray(rotated_point_cloud.colors) 


        rotated_point_cloud_list.append(xyz_coordinates)
        rotated_point_cloud_color_list.append(rgb_colors)
            
        # 生成投影图保存路径
        file_name_without_ext = os.path.splitext(filename)[0]        
        # 保存投影图
        projection_2d_save = os.path.join(projection_2d_save_path,f'{file_name_without_ext}.png')
        image_path, img_info = project_and_save_image(rotated_point_cloud, projection_2d_save)
        image_paths.append(image_path)
        img_info_list.append(img_info)
        logging.info(f"处理完成: {filename}")

    # 使用nnU-Net进行预测并裁剪点云
    # cropped_results = nnunet_predict_and_crop(
    #     predictor, image_paths, rotated_point_cloud_list, rotated_point_cloud_color_list, img_info_list, cropped_output_folder
    # )

    # 
    prjection_2d = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\nnUnet2d\JSON\linshi'
    for file in os.listdir(prjection_2d):
        if file.endswith('.png'):
            image_paths.append(os.path.join(prjection_2d,file))
    cropped_results = nnunet_predict_and_crop(
        image_paths, rotated_point_cloud_list, rotated_point_cloud_color_list, img_info_list, cropped_output_folder
    )

    logging.info("所有点云文件处理完毕并完成nnU-Net分割和裁剪")




if __name__ == "__main__":
    main()