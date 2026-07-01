# import os
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt


# def vis(pcd, file3):
#     """
#     对点云做 DBSCAN 聚类，并返回点数最多的簇
#     """
#     with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#         labels = np.array(
#             pcd.cluster_dbscan(eps=2, min_points=10, print_progress=True)
#         )

#     max_label = labels.max()
#     print(f"{file3} {max_label + 1} clusters")

#     # 可视化颜色
#     colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
#     colors[labels < 0] = 0
#     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

#     # 找到点数最多的聚类编号
#     unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
#     if len(unique_labels) == 0:
#         # 如果没有聚类，直接返回原点云
#         return pcd

#     largest_cluster_label = unique_labels[np.argmax(counts)]
#     largest_cluster_points = np.asarray(pcd.points)[labels == largest_cluster_label]
#     largest_cluster_colors = np.asarray(pcd.colors)[labels == largest_cluster_label]

#     largest_cluster_pcd = o3d.geometry.PointCloud()
#     largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
#     largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
#     o3d.visualization.draw_geometries([largest_cluster_pcd], width=800, height=800, front=[-0.4999, -0.1659, -0.8499],
#                                       window_name=f"Point Clouds: {file3}")

#     return largest_cluster_pcd


# def filter_pcd(path):
#     for folder in os.listdir(path):
#         path2 = os.path.join(path, folder)
#         if not os.path.isdir(path2):
#             continue

#         for file3 in os.listdir(path2):
#             file_path = os.path.join(path2, file3)
#             data = np.loadtxt(file_path)

#             coord = data[:, :3]
#             colors = data[:, 3:6]
#             scalar_val = float(file3.split(".")[0][-1])  # 取文件名最后一个字符作为 scalar
#             scalar_zero = np.zeros_like(data[:, 0])

#             # 过滤第 6 列 > 0 的点
#             filt_data = data[data[:, 6] > 0]
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(filt_data[:, :3])
#             pcd.colors = o3d.utility.Vector3dVector(filt_data[:, 3:6] / 255.0)

#             # DBSCAN 聚类，取最大簇
#             largest_cluster_pcd = vis(pcd, file3)
#             # largest_cluster_points = np.asarray(largest_cluster_pcd.points)
#             # largest_cluster_colors = np.asarray(largest_cluster_pcd.colors)
#             # largest_cluster_scalar = np.full((largest_cluster_points.shape[0], 1), scalar_val)
#             #
#             # # 从原始 coord 里去掉最大簇的点
#             # coord_tuples = [tuple(pt) for pt in coord]
#             # largest_cluster_set = set(tuple(pt) for pt in largest_cluster_points)

#             # remaining_points = np.array([pt for pt in coord_tuples if pt not in largest_cluster_set])
#             # colors_tuples = [tuple(c) for c in colors]
#             # remaining_colors = np.array(
#             #     [colors_tuples[i] for i, pt in enumerate(coord_tuples) if pt not in largest_cluster_set])
#             # remaining_scalar = np.zeros((remaining_points.shape[0], 1))
#             #
#             # # 合并
#             # coord_all = np.vstack([remaining_points, largest_cluster_points])
#             # colors_all = np.vstack([remaining_colors, largest_cluster_colors * 255.0])
#             # scalar_all = np.vstack([remaining_scalar, largest_cluster_scalar])
#             #
#             # data_all = np.hstack([coord_all, colors_all, scalar_all])
#             #
#             # # 保存
#             # os.makedirs(f'C:\\yuechen\\code\\jiaohuaying\\2.data\\0128\\txt\\角化龈-dbscan\\processed\\{folder}',exist_ok=True)
#             # np.savetxt(f'C:\\yuechen\\code\\jiaohuaying\\2.data\\0128\\txt\\角化龈-dbscan\\processed\\{folder}\\{file3}', data_all, fmt="%.6f")


# if __name__ == "__main__":
#     filter_pcd(r'C:\yuechen\code\jiaohuaying\2.data\0128\txt\wash\角化龈-dbscan')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


# ===================== 直接改这里 =====================
IN_ROOT = r"\\Desktop-76khoer\d\1.CY-SPACE\JiaoHuaYing\fei\newdata\0247"
OUT_ROOT = r"\\Desktop-76khoer\d\1.CY-SPACE\JiaoHuaYing\fei\newdata\0247-new"
RECURSIVE = True

FLIP_V = True

TARGET_POINTS = 1000000   # 每个 ply 固定采样 100 万点
BLACK_MEAN_THR = -1       # 如果过滤黑背景，最终点数可能少于 100 万
RANDOM_SEED = 42

DEFAULT_COLOR = (200, 200, 200)
# =====================================================


def bilinear_sample_rgb_batch(img_rgb, u, v, flip_v=True):
    h, w, _ = img_rgb.shape

    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    if flip_v:
        v = 1.0 - v

    x = u * (w - 1)
    y = v * (h - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    dx = (x - x0)[:, None].astype(np.float32)
    dy = (y - y0)[:, None].astype(np.float32)

    c00 = img_rgb[y0, x0].astype(np.float32)
    c10 = img_rgb[y0, x1].astype(np.float32)
    c01 = img_rgb[y1, x0].astype(np.float32)
    c11 = img_rgb[y1, x1].astype(np.float32)

    c0 = c00 * (1 - dx) + c10 * dx
    c1 = c01 * (1 - dx) + c11 * dx
    c = c0 * (1 - dy) + c1 * dy

    return np.clip(np.round(c), 0, 255).astype(np.uint8)


def guess_texture_path(ply_path: Path, ply_obj: PlyData):
    for c in getattr(ply_obj, "comments", []):
        s = c.strip()
        if s.lower().startswith("texturefile "):
            tex_name = s.split(" ", 1)[1].strip()
            tex_path = (ply_path.parent / tex_name).resolve()
            if tex_path.exists():
                return tex_path

    candidates = [
        ply_path.with_suffix(".png"),
        ply_path.with_suffix(".jpg"),
        ply_path.with_suffix(".jpeg"),
        Path(str(ply_path) + ".png"),
        Path(str(ply_path) + ".jpg"),
        Path(str(ply_path) + ".jpeg"),
        ply_path.parent / (ply_path.stem + ".png"),
        ply_path.parent / (ply_path.stem + ".jpg"),
        ply_path.parent / (ply_path.stem + ".jpeg"),
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    return None


def triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def sample_barycentric(n, rng):
    u = rng.random(n, dtype=np.float32)
    v = rng.random(n, dtype=np.float32)

    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]

    w = 1.0 - u - v
    return u, v, w


def write_point_cloud_ply(points, colors, out_path: Path):
    vertex_dtype = np.dtype([
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])

    arr = np.empty(len(points), dtype=vertex_dtype)
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    arr["red"] = colors[:, 0]
    arr["green"] = colors[:, 1]
    arr["blue"] = colors[:, 2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(out_path))


def collect_valid_triangles(ply, vertices):
    fdata = ply["face"].data

    face_indices = []
    face_texcoords = []
    face_areas = []

    invalid_faces = 0
    total_faces = 0

    for face in fdata:
        total_faces += 1

        if "vertex_indices" not in face.dtype.names:
            invalid_faces += 1
            continue

        vi = np.asarray(face["vertex_indices"], dtype=np.int64)

        if len(vi) != 3:
            invalid_faces += 1
            continue

        v0, v1, v2 = vertices[vi[0]], vertices[vi[1]], vertices[vi[2]]
        area = triangle_area(v0, v1, v2)

        if area <= 0:
            invalid_faces += 1
            continue

        face_indices.append(vi)
        face_areas.append(area)

        if "texcoord" in face.dtype.names:
            tc = np.asarray(face["texcoord"], dtype=np.float32)
            if tc.size == 6:
                face_texcoords.append(tc.reshape(3, 2))
            else:
                face_texcoords.append(None)
        else:
            face_texcoords.append(None)

    return face_indices, face_texcoords, np.asarray(face_areas, dtype=np.float64), total_faces, invalid_faces


def convert_mesh_to_uniform_pointcloud(in_ply: Path, out_ply: Path, rng):
    ply = PlyData.read(str(in_ply))

    if "vertex" not in ply or "face" not in ply:
        print(f"[SKIP] no vertex/face: {in_ply}")
        return False

    vdata = ply["vertex"].data
    names = vdata.dtype.names

    if not all(k in names for k in ("x", "y", "z")):
        print(f"[SKIP] vertex has no xyz: {in_ply}")
        return False

    vertices = np.stack([vdata["x"], vdata["y"], vdata["z"]], axis=1).astype(np.float32)

    has_vertex_rgb = all(k in names for k in ("red", "green", "blue"))
    if has_vertex_rgb:
        vertex_colors = np.stack(
            [vdata["red"], vdata["green"], vdata["blue"]],
            axis=1
        ).astype(np.float32)
    else:
        vertex_colors = None

    tex_path = guess_texture_path(in_ply, ply)
    tex_np = None
    if tex_path is not None:
        tex_np = np.array(Image.open(str(tex_path)).convert("RGB"), dtype=np.uint8)

    face_indices, face_texcoords, areas, total_faces, invalid_faces = collect_valid_triangles(
        ply, vertices
    )

    if len(face_indices) == 0 or areas.sum() <= 0:
        print(f"[SKIP] no valid triangle faces: {in_ply}")
        return False

    probs = areas / areas.sum()
    sample_counts = rng.multinomial(TARGET_POINTS, probs)

    sampled_points = []
    sampled_colors = []

    for vi, uv, n_samples in zip(face_indices, face_texcoords, sample_counts):
        if n_samples <= 0:
            continue

        v0, v1, v2 = vertices[vi[0]], vertices[vi[1]], vertices[vi[2]]

        u, v, w = sample_barycentric(n_samples, rng)

        pts = (
            w[:, None] * v0[None, :] +
            u[:, None] * v1[None, :] +
            v[:, None] * v2[None, :]
        )

        if tex_np is not None and uv is not None:
            uv_pts = (
                w[:, None] * uv[0][None, :] +
                u[:, None] * uv[1][None, :] +
                v[:, None] * uv[2][None, :]
            )
            cols = bilinear_sample_rgb_batch(
                tex_np,
                uv_pts[:, 0],
                uv_pts[:, 1],
                flip_v=FLIP_V,
            )
            color_mode = "texture_rgb"

        elif has_vertex_rgb:
            c0, c1, c2 = vertex_colors[vi[0]], vertex_colors[vi[1]], vertex_colors[vi[2]]
            cols = (
                w[:, None] * c0[None, :] +
                u[:, None] * c1[None, :] +
                v[:, None] * c2[None, :]
            )
            cols = np.clip(np.round(cols), 0, 255).astype(np.uint8)
            color_mode = "vertex_rgb"

        else:
            cols = np.tile(np.array(DEFAULT_COLOR, dtype=np.uint8), (n_samples, 1))
            color_mode = "default_rgb"

        if BLACK_MEAN_THR >= 0:
            keep_mask = cols.mean(axis=1) > BLACK_MEAN_THR
            pts = pts[keep_mask]
            cols = cols[keep_mask]

        if len(pts) > 0:
            sampled_points.append(pts.astype(np.float32))
            sampled_colors.append(cols.astype(np.uint8))

    if not sampled_points:
        print(f"[SKIP] sampled no points after filtering: {in_ply}")
        return False

    sampled_points = np.concatenate(sampled_points, axis=0)
    sampled_colors = np.concatenate(sampled_colors, axis=0)

    write_point_cloud_ply(sampled_points, sampled_colors, out_ply)

    print(
        f"[OK] {in_ply} -> {out_ply} | "
        f"mode=uniform_area_sampling | color={color_mode} | "
        f"faces={total_faces}, invalid_faces={invalid_faces}, "
        f"points={len(sampled_points)}"
    )

    return True


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    in_root = Path(IN_ROOT)
    out_root = Path(OUT_ROOT)

    ply_files = sorted(in_root.rglob("*.ply") if RECURSIVE else in_root.glob("*.ply"))

    if not ply_files:
        print("No .ply found")
        return

    ok, skip = 0, 0

    for p in ply_files:
        rel = p.relative_to(in_root)
        out_p = out_root / rel.parent / f"{p.stem}.ply"

        try:
            if convert_mesh_to_uniform_pointcloud(p, out_p, rng):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            skip += 1
            print(f"[ERR] {p}: {e}")

    print(f"\nDone. ok={ok}, skip_or_err={skip}, total={len(ply_files)}")


if __name__ == "__main__":
    main()
