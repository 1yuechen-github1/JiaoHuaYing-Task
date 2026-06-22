import os
import numpy as np
import shutil

# pcd_path = r"Z:\1.CY-SPACE\JiaoHuaYing\only_upper_front_teeth\xiaheya" 
# npy_path = r"Z:\1.CY-SPACE\JiaoHuaYing\only_upper_front_teeth\xiaheya_npy"

# for index in os.listdir(pcd_path):
#     total_points = 0

#     fold_path = os.path.join(pcd_path, index)
#     fold_path1 = os.path.join(npy_path, index)

#     for file in os.listdir(fold_path):
#         data = np.loadtxt(os.path.join(fold_path, file))

#         if data.ndim == 1:
#             data = data.reshape(1, -1)

#         total_points += data.shape[0]

#     npy_data = np.load(os.path.join(fold_path1, 'segment.npy'))
#     if(total_points!=npy_data.shape[0]):
#         print(index, total_points, npy_data.shape)


# Z:\1.CY-SPACE\JiaoHuaYing\1.AllData-PointCloud-QueYaQu\0002\0002_ls_upper_1_2.txt
pcd_path = r"Z:\1.CY-SPACE\JiaoHuaYing\1.AllData-PointCloud-QueYaQu"
shangqianya_dir = r'Z:\1.CY-SPACE\JiaoHuaYing\only_upper_front_teeth\shangqianya'
shanghouya_dir = r'Z:\1.CY-SPACE\JiaoHuaYing\only_upper_front_teeth\shanghouya'
xiaheya_dir = r'Z:\1.CY-SPACE\JiaoHuaYing\only_upper_front_teeth\xiaheya'
for file in os.listdir(pcd_path):
    file_dir = os.path.join(pcd_path,file)
    for file1 in os.listdir(file_dir):
        file_path = os.path.join(file_dir,file1)
        data = np.loadtxt(file_path)
        scalar = data[:, 6]
        print(file1,np.unique(scalar))
        if np.unique(scalar)[1] == 1:
            # print(file1,np.unique(scalar))
            os.makedirs(os.path.join(shangqianya_dir,file),exist_ok=True)
            shutil.copy(file_path,os.path.join(shangqianya_dir,file,file1))

        elif np.unique(scalar)[1] == 2:
            # print(file1,np.unique(scalar))
            os.makedirs(os.path.join(shanghouya_dir,file),exist_ok=True)
            shutil.copy(file_path,os.path.join(shanghouya_dir,file,file1))

        else:
            # print(file1,np.unique(scalar))
            os.makedirs(os.path.join(xiaheya_dir,file),exist_ok=True)
            shutil.copy(file_path,os.path.join(xiaheya_dir,file,file1))

