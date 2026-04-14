import os
import numpy as np

import os
import numpy as np

def genera_npy(path):
    save_root = r'C:\yuechen\code\jiaohuaying\2.data\3.0326_data\wash\6.npy'

    for file in os.listdir(path):
        all_coord = []
        all_color = []
        all_scalar = []

        file_path = os.path.join(path, file)
        if not os.path.isdir(file_path):
            continue

        for file2 in os.listdir(file_path):
            data = np.loadtxt(os.path.join(file_path, file2))
            all_coord.append(data[:, 0:3])
            all_color.append(data[:, 3:6])
            all_scalar.append(data[:, 6])

        merged_coord = np.vstack(all_coord)
        merged_color = np.vstack(all_color)
        merged_scalar = np.concatenate(all_scalar)

        n_points = merged_coord.shape[0]

        normal = np.zeros((n_points, 3), dtype=np.float32)
        instance = np.zeros(n_points, dtype=np.int32)
        segment = (merged_scalar > 0).astype(np.int32)

        save_dir = os.path.join(save_root, file)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "coord.npy"), merged_coord.astype(np.float32))
        np.save(os.path.join(save_dir, "color.npy"), merged_color.astype(np.float32))
        np.save(os.path.join(save_dir, "normal.npy"), normal)
        np.save(os.path.join(save_dir, "instance.npy"), instance)
        np.save(os.path.join(save_dir, "segment.npy"), segment)
        np.save(os.path.join(save_dir, "label.npy"), merged_scalar.astype(np.float32))




# genera_npy(r'C:\yuechen\code\jiaohuaying\2.data\3.0326_data\wash\5.缺牙区-有角化龈\2.缺牙区')


def read_npy(path):
    for file1 in os.listdir(path):
        for file in os.listdir(os.path.join(path, file1)):
            if file.endswith('label.npy'):
                data = np.load(os.path.join(path, file1,'label.npy'))
                print(file1, np.unique(data))

read_npy(r'C:\yuechen\code\jiaohuaying\2.data\3.0326_data\wash\6.npy')