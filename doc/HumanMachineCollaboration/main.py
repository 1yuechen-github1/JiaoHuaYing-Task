import open3d as o3d
import os
import numpy as np
from scipy.spatial import cKDTree
from pcd_to_mesh import PointCloudMeshConverter

def smooth_labels(points, labels, k=20, iterations=3):
    """用近邻多数投票平滑 0/1 标签。"""
    points = np.asarray(points)
    labels = np.asarray(labels).reshape(-1).copy()

    tree = cKDTree(points)
    _, neighbor_ids = tree.query(points, k=min(k + 1, len(points)))
    neighbor_ids = neighbor_ids[:, 1:]

    binary_labels = (labels >= 1).astype(np.uint8)

    for _ in range(iterations):
        neighbor_target_ratio = binary_labels[neighbor_ids].mean(axis=1)

        # 超过一半邻居是目标点，则该点设为目标；否则设为非目标
        binary_labels = (neighbor_target_ratio >= 0.5).astype(np.uint8)

    return binary_labels


def get_surface_boundary(points, labels, target_label=1, k=12):
    points = np.asarray(points)

    # 先平滑 AI 标签
    smooth_binary = smooth_labels(points, labels, k=20, iterations=4)
    target_mask = smooth_binary > 0

    tree = cKDTree(points)
    _, neighbor_ids = tree.query(points, k=min(k + 1, len(points)))
    neighbor_ids = neighbor_ids[:, 1:]

    boundary_ids = []

    for i in np.where(target_mask)[0]:
        if np.any(~target_mask[neighbor_ids[i]]):
            boundary_ids.append(i)

    return points[np.asarray(boundary_ids)]


def vis(pcd_list, file):
    o3d.visualization.draw_geometries(
        pcd_list,
        window_name=f"Point Clouds: {file}",
        width=800,
        height=600
    )


def transform_pcd(jiaohua, jiaohua_file):
    if 'upp' in jiaohua_file or 'Upp' in jiaohua_file:
        is_maxilla = True
    else:
        is_maxilla = False
    print(jiaohua_file, is_maxilla)



def visualize_missing_tooth_area(txt_path, target_label, output_dir,filename):
    # TXT: x y z r g b ai_label
    # filename = os.path.basename(txt_path)
    # 缺牙区
    data = np.loadtxt(txt_path)
    points = data[:, :3]
    gt = data[:, 6:7].astype(int).flatten()
    ai = data[:, 7:8].astype(int).flatten() 
    labels = data[:, 7:8].astype(int).flatten()

    boundary_points = get_surface_boundary(
        points,
        labels,
        target_label=target_label,
        k=12,
    )

    gt_boundary_points = get_surface_boundary(
        points,
        gt,
        target_label=target_label,
        k=12,
    )
    ai_boundary_points = get_surface_boundary(
        points,
        ai,
        target_label=target_label,
        k=12,
    )
    

    # 原始点云
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    colors = np.tile([0.2, 0.3, 1.0], (len(points), 1))
    colors[labels == target_label] = [1.0, 0.0, 0.0]
    cloud.colors = o3d.utility.Vector3dVector(colors)

    # 边界点：绿色，并放大显示
    ai_boundary = o3d.geometry.PointCloud()
    ai_boundary.points = o3d.utility.Vector3dVector(ai_boundary_points)
    ai_boundary.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 0.0], (len(ai_boundary_points), 1))
    )


    gt_boundary = o3d.geometry.PointCloud()
    gt_boundary.points = o3d.utility.Vector3dVector(gt_boundary_points)
    gt_boundary.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.0, 0.0], (len(gt_boundary_points), 1))
    )

    # 边界和原始点云叠加保存
    # 原始数据只取 xyz、rgb、label
    # 你的 label 在第 8 列，所以索引为 7
    original_data = data[:, :8]
    new_data = original_data.copy()

    # 每个原始点到最近边界点的距离
    ai_boundary_tree = cKDTree(ai_boundary_points)
    gt_boundary_tree = cKDTree(gt_boundary_points)
    ai_distances, _ = ai_boundary_tree.query(points, k=1)
    gt_distances, _ = gt_boundary_tree.query(points, k=1)

    # 依据点云单位调整；先试 0.05 或 0.1
    # ai_boundary_mask = ai_distances < 0.05
    gt_boundary_mask = gt_distances < 0.05
    red_col = [255, 0, 0]
    # green_col = [0, 255, 0]
    os.makedirs(os.path.join(output_dir, 'missing_tooth_area'), exist_ok=True)
    new_data_copy = new_data

    new_data_copy[gt_boundary_mask, 3:6] = red_col
    output_path = os.path.join(
        output_dir,
        f"missing_tooth_area\\gt_{os.path.splitext(filename)[0]}.txt",
    )
    np.savetxt(
        output_path,
        new_data_copy,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d", "%d"],
        comments="",
    )

    new_data[gt_boundary_mask, 3:6] = red_col
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(new_data[:, 3:6] / 255.0)

    # 可视化点云的ai和gt的边界线
    # vis([pcd], '')
    # 保存为点云， ai和gt叠加，仅仅保留边界线
    output_path = os.path.join(
        output_dir,
        f"missing_tooth_area\\merge_{os.path.splitext(filename)[0]}.txt",
    )
    np.savetxt(
        output_path,
        new_data,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d", "%d"],
        comments="",
    )


    return data




def visualize_oral_scan(
    oral_scan_path,
    target_label,
    output_dir,
    data,
    filename,
    distance_threshold=0.05
):
    points = data[:, :3]

    gt = data[:, 6].astype(int)
    ai = data[:, 7].astype(int)

    gt_boundary_points = get_surface_boundary(
        points,
        gt,
        target_label=target_label,
        k=12
    )

    ai_boundary_points = get_surface_boundary(
        points,
        ai,
        target_label=target_label,
        k=12
    )

    # 读取原始口扫点云
    oral_pcd = o3d.io.read_point_cloud(
        str(oral_scan_path)
    )

    oral_points = np.asarray(
        oral_pcd.points,
        dtype=np.float64
    )

    oral_colors = np.asarray(
        oral_pcd.colors,
        dtype=np.float64
    )

    # 在原始口扫点云中寻找边界对应点
    oral_tree = cKDTree(oral_points)

    ai_distances, ai_indices = oral_tree.query(
        ai_boundary_points,
        k=1
    )

    gt_distances, gt_indices = oral_tree.query(
        gt_boundary_points,
        k=1
    )

    ai_indices = ai_indices[
        ai_distances <= distance_threshold
    ]

    gt_indices = gt_indices[
        gt_distances <= distance_threshold
    ]

    ai_boundary_mask = np.zeros(
        len(oral_points),
        dtype=bool
    )

    gt_boundary_mask = np.zeros(
        len(oral_points),
        dtype=bool
    )

    ai_boundary_mask[ai_indices] = True
    gt_boundary_mask[gt_indices] = True

    # 生成三个版本的颜色
    ai_colors = oral_colors.copy()
    gt_colors = oral_colors.copy()
    merge_colors = oral_colors.copy()

    # AI 结果：只显示 AI 边界，绿色
    ai_colors[ai_boundary_mask] = [0.0, 1.0, 0.0]

    # GT 结果：只显示 GT 边界，红色
    gt_colors[gt_boundary_mask] = [1.0, 0.0, 0.0]

    # 合并结果：AI 绿色，GT 红色，重合黄色
    merge_colors[ai_boundary_mask] = [0.0, 1.0, 0.0]
    merge_colors[gt_boundary_mask] = [1.0, 0.0, 0.0]

    overlap_mask = ai_boundary_mask & gt_boundary_mask
    merge_colors[overlap_mask] = [1.0, 1.0, 0.0]

    # 创建点云
    def make_pcd(colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            oral_points
        )
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(colors, 0.0, 1.0)
        )
        return pcd

    ai_pcd = make_pcd(ai_colors)
    gt_pcd = make_pcd(gt_colors)
    merge_pcd = make_pcd(merge_colors)

    # 输出目录
    oral_output_dir = os.path.join(
        output_dir,
        "oral_scan"
    )
    os.makedirs(oral_output_dir, exist_ok=True)

    stem = os.path.splitext(filename)[0]

    ai_output_path = os.path.join(
        oral_output_dir,
        f"{stem}_ai.ply"
    )

    gt_output_path = os.path.join(
        oral_output_dir,
        f"{stem}_gt.ply"
    )

    merge_output_path = os.path.join(
        oral_output_dir,
        f"{stem}_merge.ply"
    )

    # 保存彩色点云
    # o3d.io.write_point_cloud(
    #     ai_output_path,
    #     ai_pcd,
    #     write_ascii=False
    # )

    # o3d.io.write_point_cloud(
    #     gt_output_path,
    #     gt_pcd,
    #     write_ascii=False
    # )

    # o3d.io.write_point_cloud(
    #     merge_output_path,
    #     merge_pcd,
    #     write_ascii=False
    # )

    # 转换成 Mesh
    converter = PointCloudMeshConverter(
        radius_factors=(3.5, 4.5, 6.0),
        min_component_triangles=0,
        save_normals=False
    )

    ai_mesh_path = os.path.join(
        oral_output_dir,
        f"{stem}_ai_mesh.ply"
    )

    gt_mesh_path = os.path.join(
        oral_output_dir,
        f"{stem}_gt_mesh.ply"
    )

    merge_mesh_path = os.path.join(
        oral_output_dir,
        f"{stem}_merge_mesh.ply"
    )

    # 如果类中有 build_mesh，可使用内存点云直接转换
    ai_mesh = converter.build_mesh(ai_pcd)
    gt_mesh = converter.build_mesh(gt_pcd)
    merge_mesh = converter.build_mesh(merge_pcd)

    # 保存 Mesh，不保存法向量，只保存颜色
    o3d.io.write_triangle_mesh(
        ai_mesh_path,
        ai_mesh,
        write_ascii=False,
        write_vertex_normals=False,
        write_vertex_colors=True
    )

    o3d.io.write_triangle_mesh(
        gt_mesh_path,
        gt_mesh,
        write_ascii=False,
        write_vertex_normals=False,
        write_vertex_colors=True
    )

    o3d.io.write_triangle_mesh(
        merge_mesh_path,
        merge_mesh,
        write_ascii=False,
        write_vertex_normals=False,
        write_vertex_colors=True
    )

    print("AI 点云:", ai_output_path)
    print("GT 点云:", gt_output_path)
    print("合并点云:", merge_output_path)

    print("AI Mesh:", ai_mesh_path)
    print("GT Mesh:", gt_mesh_path)
    print("合并 Mesh:", merge_mesh_path)

    return {
        "ai_pcd": ai_pcd,
        "gt_pcd": gt_pcd,
        "merge_pcd": merge_pcd,
        "ai_mesh": ai_mesh,
        "gt_mesh": gt_mesh,
        "merge_mesh": merge_mesh
    }



oral_scan_dir = r'Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\test\rotake_oral_scan'
missing_tooth_area_dir = r'Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\pointcept-加上下颌位置信息-法向量-边界做数据增强\result-txt-test'
output_dir = r'Z:\1.CY-SPACE\JiaoHuaYing\SupplementaryExperiments\核对数据后\pointcept-加上下颌位置信息-法向量-边界做数据增强\output'

for oral_scan in os.listdir(oral_scan_dir):
    oral_scan_path = os.path.join(oral_scan_dir,oral_scan)
    print('oral_scan_path:',oral_scan_path)
    for missing_tooth in os.listdir(missing_tooth_area_dir):
        print(oral_scan, oral_scan)
        if missing_tooth[:-8] in oral_scan:
            missing_tooth_area_path = os.path.join(missing_tooth_area_dir,missing_tooth)
            output_path = os.path.join(output_dir,missing_tooth)
            
            data = visualize_missing_tooth_area(missing_tooth_area_path,1,output_dir,missing_tooth)
            output_path = os.path.join(output_dir,oral_scan)
            # visualize_oral_scan(oral_scan_path,1,output_dir,data,oral_scan)



