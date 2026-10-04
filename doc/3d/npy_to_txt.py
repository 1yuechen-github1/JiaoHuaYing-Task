

import os
import numpy as np

def npy_to_txt(path):
    data_dic = {}
    for file in os.listdir(path):
        data = np.load(os.path.join(path, file))
        index = file.split('.')[0][0:4]
        data_dic[index] = data.reshape(-1, 1)  # 确保是列
    return data_dic


def read_txt(data_dic):
    
    path = r'Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\test\linshi\queyaqu'
    path2 = r'Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\test\linshi'
    os.makedirs(path2, exist_ok=True)

    for index, pred_all in data_dic.items():
        txt_dir = os.path.join(path, index)

        start = 0  # pred 游标

        for file in os.listdir(txt_dir):
            data = np.loadtxt(os.path.join(txt_dir, file))

            # 🔥 保证 data 是二维
            if data.ndim == 1:
                data = data.reshape(1, -1)

            points = np.vstack(data)
            point_len = len(points)

            end = start + point_len

            if end > len(pred_all):
                raise ValueError(
                    f'{index} pred 不够用：需要 {end}，只有 {len(pred_all)}'
                )

            pred = pred_all[start:end]
            # pred = pred_all[start:end].ravel()

            start = end

            output_data = np.hstack([points, pred])
            # output_data = points.copy()
            # output_data[:, -1] = pred

            save_path = os.path.join(path2, f'{file}')
            np.savetxt(save_path, output_data, fmt='%.6f')

            # ✅ 输出列数
            num_cols = output_data.shape[1]
            print(f"{file} -> 列数: {num_cols}")

        # 可选：检查是否刚好用完
        if start != len(pred_all):
            print(f"⚠️ {index} pred 没用完: 用了 {start} / 总共 {len(pred_all)}")


data = npy_to_txt(r'Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\test\linshi\npy')
read_txt(data)


# import os
# import shutil

# path = r'Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\MaxillaryInformation\MaxillaryInformation'
# path2 = r'Y:\1.CY-SPACE\JiaoHuaYing\1.AllData-PointCloud-QueYaQu'
# path3 = r'Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\queyaqu'

# os.makedirs(path3, exist_ok=True)

# for file in os.listdir(path):
#     index = file[0:4]

#     src = os.path.join(path2, index)
#     dst = os.path.join(path3, index)

#     if os.path.isdir(src):
#         shutil.copytree(src, dst, dirs_exist_ok=True)
#         print("复制文件夹:", src, "->", dst)
#     else:
#         print("不存在或不是文件夹:", src)