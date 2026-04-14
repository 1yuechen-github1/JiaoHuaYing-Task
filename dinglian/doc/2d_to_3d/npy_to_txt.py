

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
    path = r'C:\Users\yuechen\Desktop\result\test'
    path2 = r'C:\Users\yuechen\Desktop\result\txt'
    os.makedirs(path2, exist_ok=True)

    for index, pred_all in data_dic.items():
        txt_dir = os.path.join(path, index)

        start = 0  # 🔥 pred 游标

        for file in os.listdir(txt_dir):
            data = np.loadtxt(os.path.join(txt_dir, file))
            points = np.vstack(data)
            point_len = len(points)

            end = start + point_len

            if end > len(pred_all):
                raise ValueError(
                    f'{index} pred 不够用：需要 {end}，只有 {len(pred_all)}'
                )

            pred = pred_all[start:end]
            start = end  # 🔥 移动游标

            print(index, file, point_len, pred.shape)

            output_data = np.hstack([points, pred])
            np.savetxt(
                os.path.join(path2, f'{file}'),
                output_data,
                fmt='%.6f'
            )


data = npy_to_txt(r'C:\Users\yuechen\Desktop\result\npy')
read_txt(data)
