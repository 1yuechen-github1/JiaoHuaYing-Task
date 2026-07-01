# import os
# import numpy as np
# from scipy.spatial import cKDTree

# input_root = r"Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\MaxillaryInformation\MaxillaryInformation"
# output_root = r"Y:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\MaxillaryInformation-gradient"

# os.makedirs(output_root, exist_ok=True)

# k = 16
# boundary_radius = 0.8
# top_ratio = 0.2

# for filename in os.listdir(input_root):
#     if not filename.endswith(".txt"):
#         continue

#     path = os.path.join(input_root, filename)
#     data = np.loadtxt(path)

#     coord = data[:, 0:3]
#     raw_color = data[:, 3:6].copy()
#     color = raw_color.astype(np.float32)

#     gt = data[:, 6].astype(np.int64)
#     ai = data[:, 7].astype(np.int64)

#     # 颜色只用于计算梯度，保存时仍使用 0~255
#     if color.max() > 1.5:
#         color = color / 255.0

#     tree = cKDTree(coord)

#     # 1. 找每个点的 kNN
#     _, knn_idx = tree.query(coord, k=k + 1)
#     knn_idx = knn_idx[:, 1:]

#     # 2. 找 AI 预测边界点：邻居中存在不同 ai label
#     ai_neighbor = ai[knn_idx]
#     ai_boundary = (ai_neighbor != ai[:, None]).any(axis=1)

#     if ai_boundary.sum() == 0:
#         print(f"{filename}: no ai boundary found, save original")
#         out_path = os.path.join(output_root, filename)
#         np.savetxt(out_path, data[:, 0:8], fmt="%.6f")
#         continue

#     boundary_coord = coord[ai_boundary]

#     # 3. 找 AI 边界附近的一圈点
#     boundary_tree = cKDTree(boundary_coord)
#     dist_to_boundary, _ = boundary_tree.query(coord, k=1)

#     boundary_band = dist_to_boundary < boundary_radius

#     # 4. 计算颜色梯度
#     color_diff = color[knn_idx] - color[:, None, :]
#     color_grad = np.linalg.norm(color_diff, axis=-1).mean(axis=1)

#     # 5. 只在 AI 边界附近取颜色梯度最大的点
#     candidate_idx = np.where(boundary_band)[0]

#     color_boundary_mask = np.zeros(len(coord), dtype=bool)

#     if len(candidate_idx) > 0:
#         candidate_grad = color_grad[candidate_idx]
#         threshold = np.quantile(candidate_grad, 1.0 - top_ratio)

#         selected_idx = candidate_idx[candidate_grad >= threshold]
#         color_boundary_mask[selected_idx] = True
#     else:
#         print(f"{filename}: no boundary band points")

#     # 6. 可视化：边界点染红，其他点保持原颜色
#     vis_color = raw_color.copy()
#     vis_color[color_boundary_mask] = np.array([255, 0, 0])

#     # 7. 保存完整点云，点数量不变
#     # 保存格式: x y z r g b gt ai
#     save_data = np.hstack(
#         [
#             coord,
#             vis_color,
#             gt[:, None],
#             ai[:, None],
#         ]
#     )

#     out_path = os.path.join(output_root, filename)
#     np.savetxt(out_path, save_data, fmt="%.6f")

#     print(
#         filename,
#         "points:", len(coord),
#         "ai_boundary:", int(ai_boundary.sum()),
#         "boundary_band:", int(boundary_band.sum()),
#         "color_boundary:", int(color_boundary_mask.sum()),
#         "saved:", len(save_data),
#     )


import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

# ============================
# 参数
# ============================
input_txt = r"C:\yuechen\code\jiaohuaying\1.code\0110\0006_lrq_upper_1_1 - Cloud.txt"
output_txt = "10006_lrq_upper_1_1.txt"
k = 16

# ============================
# 读取数据
# x y z r g b label
# ============================
data = np.loadtxt(input_txt)

xyz = data[:, :3]
rgb = data[:, 3:6]
label = data[:, 6]

# 如果RGB是0~255，归一化
if rgb.max() > 1:
    rgb = rgb / 255.0

# ============================
# 建KDTree
# ============================
tree = KDTree(xyz)

# 每个点找K近邻
_, idx = tree.query(xyz, k=k)

color_gradient = np.zeros(len(xyz))

# ============================
# 计算Local Color Variation
# ============================
for i in range(len(xyz)):

    neighbors = rgb[idx[i]]

    diff = np.linalg.norm(
        neighbors - rgb[i],
        axis=1
    )

    color_gradient[i] = diff.mean()

# 归一化
color_gradient = (color_gradient - color_gradient.min()) / (
    color_gradient.max() - color_gradient.min() + 1e-8
)

# ============================
# 保存
# ============================
out = np.column_stack([
    xyz,
    rgb,
    color_gradient,
    label
])

np.savetxt(
    output_txt,
    out,
    fmt="%.6f"
)

print("saved:", output_txt)

# ============================
# Open3D可视化
# 红=颜色变化大
# 蓝=颜色变化小
# ============================

vis_color = np.zeros((len(xyz),3))
vis_color[:,0] = color_gradient          # Red
vis_color[:,2] = 1-color_gradient        # Blue

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(vis_color)

o3d.visualization.draw_geometries([pcd])