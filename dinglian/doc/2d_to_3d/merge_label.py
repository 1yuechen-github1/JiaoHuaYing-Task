import os
import re
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
def merge_label(path):
    path2 = r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\wash\rota'
    out_dir = r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\wash'
    os.makedirs(out_dir, exist_ok=True)
    pattern = re.compile(r'^[^A-Za-z]*\.txt$')
    for file in os.listdir(path):
        m = re.match(r'^(\d+)', file)
        if not m:
            continue
        print('m:',m)
        num = m.group(1)
        # scalar_value = int(file.split('.')[0])
        # ---------- 读取 points + colors（txt / npy） ----------
        for file4 in os.listdir(os.path.join(path,num)):
            data = np.loadtxt(os.path.join(path,num,file4))
            points = data[:, :3]
            colors = data[:, 3:6]
            N = points.shape[0]
            scalar_value = int(file4.split('.')[0][-1])
            print('file:', file, scalar_value)
            # ---------- 默认 scalar = 0 ----------
            scalar_col = np.zeros((N, 1), dtype=np.int32)

            # ---------- 找 mask ply ----------
            mask_ply_dir = os.path.join(path2, num)
            for file2 in os.listdir(mask_ply_dir):
                la_poin = np.loadtxt(os.path.join(mask_ply_dir, file2))
                la_poin = np.asarray(la_poin)
                print("num;",num)
                tree = cKDTree(la_poin)
                dist, idx = tree.query(points, k=1)
                # 距离阈值（可按你的数据尺度调）
                mask = dist < 1
                print('mask:',mask.shape,np.sum(mask))
                scalar_col[mask] = scalar_value
            out_data = np.hstack([
                points,
                colors,
                scalar_col
            ])

            out_name = f"{file4}"
            os.makedirs(os.path.join(out_dir,num), exist_ok=True)
            out_path = os.path.join(out_dir, num,out_name)

            np.savetxt(
                out_path,
                out_data,
                fmt="%.6f %.6f %.6f %.6f %.6f %.6f %d"
            )
            print("保存完成:", out_path)


if __name__ == "__main__":
    merge_label(r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\wash\11')

