import os
import open3d as o3d
from utils import rotate_point_cloud,nearest_distance_mask,keep_largest_connected_component
import numpy as np


DISTANCE_THRESHOLD = 0.30
CONNECT_RADIUS = 0.60
MIN_COMPONENT_POINTS = 100

keratinized_gums_dir = r'E:\CY\JHY\JHY_HumanVsMachineMatch\角化龈分割结果_zyy'
missing_teeth_area_dir = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\1.AllData-MissingToothArea\cropped_debug'
# log_txt = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\1.AllData-MissingToothArea\log.txt'
output_dir = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\1.AllData-MissingToothArea\MissingToothArea'

# 角化龈files
# keratinized_gums_list = sorted(
#     file
#     for root, _, files in os.walk(keratinized_gums_dir)
#     for file in files
#     # if file.lower().endswith(".ply") and len(file) < 13 
#     if 'upp' not in file and 'Upp' not in file and 'low' not in file and 'Low' not in file 
# )

data_list = ['0002', '0004', '0010', '0010', '0011', '0013', '0017', '0019', '0021', '0024', '0031', 
             '0036', '0039', '0043', '0043', '0047', '0049', '0054', '0055', '0059', '0061', '0065', 
             '0068', '0073','0075', '0078', '0081', '0081', '0086', '0187', '0365', '0372', '0375', 
             '0375', '0380']

keratinized_gums_files = sorted(
    file
    for patient_id in data_list
    for file in os.listdir(
        os.path.join(keratinized_gums_dir, patient_id)
    )
    if (
        "upp" not in file.lower()
        and "low" not in file.lower()
        and file.lower().endswith(".ply")
    )
)

# print(keratinized_gums_files)

# 缺牙区files
missing_tooth_area_list = sorted(
    file
    for root, _, files in os.walk(missing_teeth_area_dir)
    for file in files
    if file.lower().endswith(".txt")
)

print(len(data_list),len(missing_tooth_area_list))
unmatched_keratinized_gums = []
for keratinized_gums_file in data_list:
    patient_id = keratinized_gums_file[0:4]
    
    for keratinized_gums_file1 in os.listdir(os.path.join(keratinized_gums_dir,keratinized_gums_file)):
        is_maxilla = keratinized_gums_file1[-6:-4]
        if 'Upp' not in keratinized_gums_file1 and 'upp' not in keratinized_gums_file1 and 'low' not in keratinized_gums_file1 and 'Low' not in keratinized_gums_file1: 
            keratinized_gums_path = os.path.join(keratinized_gums_dir,patient_id,keratinized_gums_file1)

            pcd = o3d.io.read_point_cloud(keratinized_gums_path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            if colors is None or len(colors) != len(points):
                colors = np.ones((len(points), 3), dtype=float)
            if len(colors) > 0 and colors.max() <= 1.0:
                colors = colors * 255.0
            data = np.hstack([points, colors])
            print('keratinized_gums_file:',keratinized_gums_file, is_maxilla)
            if is_maxilla.isdigit() and int(is_maxilla) < 30:
                data = rotate_point_cloud(data, "y", 180)
                data = rotate_point_cloud(data, "x", -35)
                keratinized_gums = rotate_point_cloud(data, "z", 180)
            else:
                keratinized_gums = rotate_point_cloud(data, "x", -20)

            candidate = []
            missing_tooth_area_file_copy = None
            for missing_tooth_area_file in missing_tooth_area_list:
                if not missing_tooth_area_file.startswith(patient_id):
                    continue
                missing_tooth_area_file_copy = missing_tooth_area_file
                # missing_tooth_area_path = os.path.join(missing_teeth_area_dir,patient_id,missing_tooth_area_file)
                missing_tooth_area_path = os.path.join(missing_teeth_area_dir,missing_tooth_area_file)
                missing_tooth_area = np.loadtxt(missing_tooth_area_path)
                label = int(missing_tooth_area_file[-5:-4])

                print(keratinized_gums_file1,missing_tooth_area_file)
                print(keratinized_gums_path)
                print(missing_tooth_area_path)
                candidate_mask, distances = nearest_distance_mask(
                    missing_tooth_area, keratinized_gums, distance_threshold=DISTANCE_THRESHOLD
                )
                mask, component_count, largest_component_size = keep_largest_connected_component(
                    missing_tooth_area,
                    candidate_mask,
                    connect_radius=CONNECT_RADIUS,
                    min_component_points=MIN_COMPONENT_POINTS,
                )

                labels = np.zeros(missing_tooth_area.shape[0], dtype=int)
                labels[mask] = label
                candidate.append({
                    "file": missing_tooth_area_file,
                    "points": missing_tooth_area,
                    "labels": labels,
                    "positive_count": np.count_nonzero(labels > 0),
                })

            # print(candidate)
            # 所有缺牙区文件处理完后，选阳性点数量最多的一个
            if candidate:
                best_candidate = max(candidate, key=lambda item: item["positive_count"])
                missing_tooth_area = best_candidate["points"]
                labels = best_candidate["labels"]
                output = np.column_stack((
                    missing_tooth_area[:, :6],
                    labels
                ))
                filename = best_candidate['file']
                save_path = os.path.join(output_dir,filename)
                if os.path.exists(save_path):
                    print(save_path,'已经存在')
                    continue
                else:
                    np.savetxt(save_path,output,fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d"],)
                    print(
                        f"选择: {best_candidate['file']} | "
                        f"GT 点数: {best_candidate['positive_count']}"
                    )
            else:
                print(f"{patient_id} 没有匹配的缺牙区文件")


            # with open(log_txt, "w", encoding="utf-8") as f:
            #     f.write("以下角化龈文件未匹配到缺牙区文件：\n")
            #     for file in unmatched_keratinized_gums:
            #         f.write(f"{file}\n")

            # print(f"未匹配角化龈文件数：{len(unmatched_keratinized_gums)}")