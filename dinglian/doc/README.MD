### 2.1 对应缺牙区和对应角化龈
```python
import os
import re
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def merge_label(path):
    path2 = r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\角化龈(无rgb)'
    out_dir = r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\缺牙区(有角化龈)'
    os.makedirs(out_dir, exist_ok=True)

    pattern = re.compile(r'^[^A-Za-z]*\.txt$')

    for file in os.listdir(path):
        m = re.match(r'^(\d+)', file)
        if not m:
            continue
        print('m:',m)
        num = m.group(1)
        scalar_value = int(file.split('.')[0][-1])

        # ---------- 读取 points + colors（txt / npy） ----------
        for file4 in os.listdir(os.path.join(path,num)):
            data = np.loadtxt(os.path.join(path,num,file4))
            points = data[:, :3]
            colors = data[:, 3:6]
            N = points.shape[0]
            print(file,scalar_value)

            # ---------- 默认 scalar = 0 ----------
            scalar_col = np.zeros((N, 1), dtype=np.int32)

            # ---------- 找 mask ply ----------
            mask_ply_dir = os.path.join(path2, num)
            for file2 in os.listdir(mask_ply_dir):
                la_poin = np.loadtxt(os.path.join(mask_ply_dir, file2))
                la_poin = np.asarray(la_poin)
                print("num;",num)

                # ---------- KDTree 匹配 ----------
                tree = cKDTree(la_poin)
                dist, idx = tree.query(points, k=1)

                # 距离阈值（可按你的数据尺度调）
                mask = dist < 1
                print('mask:',mask.shape,np.sum(mask))
                scalar_col[mask] = scalar_value

            # ---------- 拼接并保存 ----------
            out_data = np.hstack([
                points,
                colors,
                scalar_col
            ])

            out_name = f"{file4}.txt"
            os.makedirs(os.path.join(out_dir,num), exist_ok=True)
            out_path = os.path.join(out_dir, num,out_name)

            np.savetxt(
                out_path,
                out_data,
                fmt="%.6f %.6f %.6f %.6f %.6f %.6f %d"
            )

            print("保存完成:", out_path)


if __name__ == "__main__":
    merge_label(r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\缺牙区')


```

### 2.2 使用dbscan分出多余的角化龈

```python
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def vis(pcd, file3):
    """
    对点云做 DBSCAN 聚类，并返回点数最多的簇
    """
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=True)
        )

    max_label = labels.max()
    print(f"{file3} {max_label + 1} clusters")

    # 可视化颜色
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 找到点数最多的聚类编号
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        # 如果没有聚类，直接返回原点云
        return pcd

    largest_cluster_label = unique_labels[np.argmax(counts)]
    largest_cluster_points = np.asarray(pcd.points)[labels == largest_cluster_label]
    largest_cluster_colors = np.asarray(pcd.colors)[labels == largest_cluster_label]

    largest_cluster_pcd = o3d.geometry.PointCloud()
    largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
    largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
    o3d.visualization.draw_geometries([largest_cluster_pcd], width=800, height=800, front=[-0.4999, -0.1659, -0.8499],
                                      window_name=f"Point Clouds: {file3}")

    return largest_cluster_pcd


def filter_pcd(path):
    for folder in os.listdir(path):
        path2 = os.path.join(path, folder)
        if not os.path.isdir(path2):
            continue

        for file3 in os.listdir(path2):
            file_path = os.path.join(path2, file3)
            data = np.loadtxt(file_path)

            coord = data[:, :3]
            colors = data[:, 3:6]
            scalar_val = float(file3.split(".")[0][-1])  # 取文件名最后一个字符作为 scalar
            scalar_zero = np.zeros_like(data[:, 0])

            # 过滤第 6 列 > 0 的点
            filt_data = data[data[:, 6] > 0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filt_data[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(filt_data[:, 3:6] / 255.0)

            # DBSCAN 聚类，取最大簇
            largest_cluster_pcd = vis(pcd, file3)
            largest_cluster_points = np.asarray(largest_cluster_pcd.points)
            largest_cluster_colors = np.asarray(largest_cluster_pcd.colors)
            largest_cluster_scalar = np.full((largest_cluster_points.shape[0], 1), scalar_val)

            # 从原始 coord 里去掉最大簇的点
            coord_tuples = [tuple(pt) for pt in coord]
            largest_cluster_set = set(tuple(pt) for pt in largest_cluster_points)

            remaining_points = np.array([pt for pt in coord_tuples if pt not in largest_cluster_set])
            colors_tuples = [tuple(c) for c in colors]
            remaining_colors = np.array(
                [colors_tuples[i] for i, pt in enumerate(coord_tuples) if pt not in largest_cluster_set])
            remaining_scalar = np.zeros((remaining_points.shape[0], 1))

            # 合并
            coord_all = np.vstack([remaining_points, largest_cluster_points])
            colors_all = np.vstack([remaining_colors, largest_cluster_colors * 255.0])
            scalar_all = np.vstack([remaining_scalar, largest_cluster_scalar])

            data_all = np.hstack([coord_all, colors_all, scalar_all])

            # 保存
            os.makedirs(f'C:\\yuechen\\code\\jiaohuaying\\2.data\\0128\\txt\\角化龈-dbscan\\processed\\{folder}',exist_ok=True)
            np.savetxt(f'C:\\yuechen\\code\\jiaohuaying\\2.data\\0128\\txt\\角化龈-dbscan\\processed\\{folder}\\{file3}', data_all, fmt="%.6f")


if __name__ == "__main__":
    filter_pcd(r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\角化龈-dbscan')

```

