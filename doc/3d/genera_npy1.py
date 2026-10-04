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

    return np.asarray(pcd.normals, dtype=np.float32)


def genera_npy(path):
    save_root = r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\1.AllData-MissingToothArea\MissingToothAreaNpy'

    for folder_name in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        print(f"\n正在处理文件夹: {folder_name}")

        txt_files = sorted(
            file_name for file_name in os.listdir(folder_path)
            if file_name.lower().endswith(".txt")
        )

        if not txt_files:
            print("  未找到 TXT 文件，跳过。")
            continue

        for instance_id, txt_file in enumerate(txt_files, start=1):
            txt_path = os.path.join(folder_path, txt_file)
            txt_name = os.path.splitext(txt_file)[0]

            print(f"  [{instance_id}/{len(txt_files)}] 读取: {txt_file}")

            try:
                data = np.loadtxt(txt_path)

                if data.ndim == 1:
                    data = data.reshape(1, -1)

                if data.shape[1] < 7:
                    print(f"  跳过: {txt_file}，列数不足 7 列。")
                    continue

                coord = data[:, 0:3].astype(np.float32)
                color = data[:, 3:6].astype(np.float32)
                scalar = data[:, 6].astype(np.float32)

                normal = estimate_normals(coord, radius=1.0, max_nn=30)

                # 每个 TXT 内的所有点属于同一个实例。
                instance = np.full(coord.shape[0], instance_id, dtype=np.int32)
                segment = (scalar > 0).astype(np.int32)

                # 每个 TXT 创建独立目录，避免同名 npy 被覆盖。
                save_dir = os.path.join(save_root, txt_name)
                os.makedirs(save_dir, exist_ok=True)

                np.save(os.path.join(save_dir, "coord.npy"), coord)
                np.save(os.path.join(save_dir, "color.npy"), color)
                np.save(os.path.join(save_dir, "normal.npy"), normal)
                np.save(os.path.join(save_dir, "instance.npy"), instance)
                np.save(os.path.join(save_dir, "segment.npy"), segment)
                np.save(os.path.join(save_dir, "label.npy"), scalar)

                print(
                    f"  已保存: {save_dir} "
                    f"(点数: {coord.shape[0]})"
                )

            except Exception as error:
                print(f"  处理失败: {txt_file}")
                print(f"  错误信息: {error}")

    print("\n全部处理完成。")


genera_npy(r'Z:\1.CY-SPACE\JiaoHuaYing\After3DVerification\1.AllData-MissingToothArea\MissingToothArea')