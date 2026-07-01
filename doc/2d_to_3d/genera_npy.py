# import os
# import numpy as np

# import os
# import numpy as np

# def genera_npy(path):
#     save_root = r'Z:\1.CY-SPACE\JiaoHuaYing\only_upper_front_teeth\shangqianya_npy'

#     for file in os.listdir(path):
#         print(file)
#         all_coord = []
#         all_color = []
#         all_scalar = []

#         file_path = os.path.join(path, file)
#         if not os.path.isdir(file_path):
#             continue

#         for file2 in os.listdir(file_path):
#             data = np.loadtxt(os.path.join(file_path, file2))
#             all_coord.append(data[:, 0:3])
#             all_color.append(data[:, 3:6])
#             all_scalar.append(data[:, 6])

#         merged_coord = np.vstack(all_coord)
#         merged_color = np.vstack(all_color)
#         merged_scalar = np.concatenate(all_scalar)

#         n_points = merged_coord.shape[0]

#         normal = np.zeros((n_points, 3), dtype=np.float32)
#         instance = np.zeros(n_points, dtype=np.int32)
#         segment = (merged_scalar > 0).astype(np.int32)

#         save_dir = os.path.join(save_root, file)
#         os.makedirs(save_dir, exist_ok=True)

#         np.save(os.path.join(save_dir, "coord.npy"), merged_coord.astype(np.float32))
#         np.save(os.path.join(save_dir, "color.npy"), merged_color.astype(np.float32))
#         np.save(os.path.join(save_dir, "normal.npy"), normal)
#         np.save(os.path.join(save_dir, "instance.npy"), instance)
#         np.save(os.path.join(save_dir, "segment.npy"), segment)
#         np.save(os.path.join(save_dir, "label.npy"), merged_scalar.astype(np.float32))


# genera_npy(r'Z:\1.CY-SPACE\JiaoHuaYing\only_upper_front_teeth\shangqianya')


import os
import numpy as np
import open3d as o3d


def estimate_normals(coord, radius=1.0, max_nn=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )

    pcd.orient_normals_consistent_tangent_plane(30)

    normal = np.asarray(pcd.normals, dtype=np.float32)
    return normal


def genera_npy(path):
    save_root = r'Y:\1.CY-SPACE\JiaoHuaYing\1.AllData-PointCloud-QueYaQu-npy'

    for file in os.listdir(path):
        print(file)

        all_coord = []
        all_color = []
        all_scalar = []
        all_instance = []

        file_path = os.path.join(path, file)
        if not os.path.isdir(file_path):
            continue

        txt_files = sorted(os.listdir(file_path))

        for instance_id, file2 in enumerate(txt_files, start=1):
            txt_path = os.path.join(file_path, file2)

            if not os.path.isfile(txt_path):
                continue

            data = np.loadtxt(txt_path)

            if data.ndim == 1:
                data = data.reshape(1, -1)

            coord = data[:, 0:3]
            color = data[:, 3:6]
            scalar = data[:, 6]

            all_coord.append(coord)
            all_color.append(color)
            all_scalar.append(scalar)

            # instance = np.full(coord.shape[0], instance_id, dtype=np.int32)
            # all_instance.append(instance)

        if len(all_coord) == 0:
            continue

        merged_coord = np.vstack(all_coord)
        merged_color = np.vstack(all_color)
        merged_scalar = np.concatenate(all_scalar)
        # instance = np.concatenate(all_instance)
        instance = np.zeros(merged_coord.shape[0], dtype=np.int32)

        normal = estimate_normals(merged_coord, radius=1.0, max_nn=30)
        segment = (merged_scalar > 0).astype(np.int32)

        save_dir = os.path.join(save_root, file)
        os.makedirs(save_dir, exist_ok=True)

        # np.save(os.path.join(save_dir, "coord.npy"), merged_coord.astype(np.float32))
        # np.save(os.path.join(save_dir, "color.npy"), merged_color.astype(np.float32))
        np.save(os.path.join(save_dir, "normal.npy"), normal.astype(np.float32))
        # np.save(os.path.join(save_dir, "instance.npy"), instance.astype(np.int32))
        # np.save(os.path.join(save_dir, "segment.npy"), segment.astype(np.int32))
        # np.save(os.path.join(save_dir, "label.npy"), merged_scalar.astype(np.float32))


genera_npy(r'Y:\1.CY-SPACE\JiaoHuaYing\1.AllData-PointCloud-QueYaQu')

