import os
import open3d as o3d
import numpy as np


def rotate_point_cloud(point_cloud, axis="y", angle_deg=0):
    point_cloud = np.asarray(point_cloud, dtype=float)
    point_cloud = np.atleast_2d(point_cloud)
    point_cloud = point_cloud[:, :3]

    angle_rad = np.radians(angle_deg)

    if axis == "x":
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([angle_rad, 0, 0])
    elif axis == "y":
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle_rad, 0])
    else:
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])

    return point_cloud @ rot.T


def nearest_distance_mask(source_points, target_points, distance_threshold):
    """返回每个 source 点到 target 点云最近点的距离及距离初筛掩码。"""
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points[:, :3])

    kdtree = o3d.geometry.KDTreeFlann(target_pcd)

    distances = np.full(source_points.shape[0], np.inf, dtype=np.float64)

    for idx, point in enumerate(source_points[:, :3]):
        count, _, squared_distances = kdtree.search_knn_vector_3d(point, 1)
        if count > 0:
            distances[idx] = np.sqrt(squared_distances[0])

    return distances <= distance_threshold, distances


def keep_largest_connected_component(points, candidate_mask, connect_radius,
                                     min_component_points=1):
    """在初筛阳性点中保留最大空间连通块，消除背景散块。"""
    candidate_indices = np.flatnonzero(candidate_mask)
    result_mask = np.zeros(len(points), dtype=bool)

    if len(candidate_indices) == 0:
        return result_mask, 0, 0

    candidate_xyz = points[candidate_indices, :3]
    candidate_pcd = o3d.geometry.PointCloud()
    candidate_pcd.points = o3d.utility.Vector3dVector(candidate_xyz)
    kdtree = o3d.geometry.KDTreeFlann(candidate_pcd)

    visited = np.zeros(len(candidate_indices), dtype=bool)
    largest_component = np.empty(0, dtype=np.int64)
    component_count = 0

    for start in range(len(candidate_indices)):
        if visited[start]:
            continue

        component_count += 1
        visited[start] = True
        stack = [start]
        component = []

        while stack:
            current = stack.pop()
            component.append(current)
            _, neighbors, _ = kdtree.search_radius_vector_3d(
                candidate_xyz[current], connect_radius
            )
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        if len(component) > len(largest_component):
            largest_component = np.asarray(component, dtype=np.int64)

    if len(largest_component) >= min_component_points:
        result_mask[candidate_indices[largest_component]] = True

    return result_mask, component_count, len(largest_component)

