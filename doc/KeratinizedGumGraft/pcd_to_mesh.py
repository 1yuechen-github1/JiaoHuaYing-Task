from pathlib import Path

import numpy as np
import open3d as o3d


class PointCloudMeshConverter:
    """使用 Ball Pivoting 将彩色点云转换为彩色 Mesh。"""

    def __init__(
        self,
        normal_radius=2.0,
        normal_max_nn=50,
        normal_consistency=30,
        radius_factors=(3.5, 4.5, 6.0),
        remove_non_manifold_edges=False,
        min_component_triangles=0,
        save_normals=False,
        verbose=True
    ):
        self.normal_radius = normal_radius
        self.normal_max_nn = normal_max_nn
        self.normal_consistency = normal_consistency
        self.radius_factors = radius_factors
        self.remove_non_manifold_edges = remove_non_manifold_edges
        self.min_component_triangles = min_component_triangles
        self.save_normals = save_normals
        self.verbose = verbose

    def _log(self, message):
        if self.verbose:
            print(message)

    @staticmethod
    def transfer_colors_to_mesh(mesh, pcd):
        """将最近点云点的 RGB 颜色复制到 Mesh 顶点。"""
        if not pcd.has_colors():
            raise ValueError("输入点云没有颜色")

        point_colors = np.asarray(
            pcd.colors,
            dtype=np.float64
        )
        mesh_vertices = np.asarray(
            mesh.vertices,
            dtype=np.float64
        )

        kdtree = o3d.geometry.KDTreeFlann(pcd)

        mesh_colors = np.empty(
            (len(mesh_vertices), 3),
            dtype=np.float64
        )

        for index, vertex in enumerate(mesh_vertices):
            count, indices, _ = (
                kdtree.search_knn_vector_3d(vertex, 1)
            )

            if count > 0:
                mesh_colors[index] = point_colors[indices[0]]
            else:
                mesh_colors[index] = [1.0, 1.0, 1.0]

        mesh_colors = np.clip(mesh_colors, 0.0, 1.0)

        mesh.vertex_colors = o3d.utility.Vector3dVector(
            mesh_colors
        )

        return mesh

    @staticmethod
    def transfer_normals_to_mesh(mesh, pcd):
        """将最近点云点的法向量复制到 Mesh 顶点。"""
        if not pcd.has_normals():
            raise ValueError("输入点云没有法向量")

        point_normals = np.asarray(
            pcd.normals,
            dtype=np.float64
        )
        mesh_vertices = np.asarray(
            mesh.vertices,
            dtype=np.float64
        )

        kdtree = o3d.geometry.KDTreeFlann(pcd)

        mesh_normals = np.empty(
            (len(mesh_vertices), 3),
            dtype=np.float64
        )

        for index, vertex in enumerate(mesh_vertices):
            count, indices, _ = (
                kdtree.search_knn_vector_3d(vertex, 1)
            )

            if count > 0:
                mesh_normals[index] = point_normals[indices[0]]
            else:
                mesh_normals[index] = [0.0, 0.0, 1.0]

        lengths = np.linalg.norm(
            mesh_normals,
            axis=1,
            keepdims=True
        )

        mesh_normals /= np.maximum(lengths, 1e-12)

        mesh.vertex_normals = o3d.utility.Vector3dVector(
            mesh_normals
        )

        return mesh

    def prepare_point_cloud(self, pcd):
        """估计并统一点云法向量。"""
        if pcd.is_empty():
            raise ValueError("点云为空")

        if not pcd.has_colors():
            raise ValueError("输入点云没有 RGB 颜色")

        self._log(f"点云数量: {len(pcd.points)}")
        self._log("正在估计点云法向量……")

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn
            )
        )

        self._log("正在统一点云法向量方向……")

        pcd.orient_normals_consistent_tangent_plane(
            self.normal_consistency
        )
        pcd.normalize_normals()

        return pcd

    def calculate_bpa_radii(self, pcd):
        """根据点云中位点间距生成 BPA 半径。"""
        distances = np.asarray(
            pcd.compute_nearest_neighbor_distance(),
            dtype=np.float64
        )

        mean_distance = float(np.mean(distances))
        median_distance = float(np.median(distances))

        radii_values = [
            median_distance * factor
            for factor in self.radius_factors
        ]

        self._log(f"平均点间距: {mean_distance}")
        self._log(f"中位点间距: {median_distance}")
        self._log(f"BPA 半径: {radii_values}")

        return radii_values

    def remove_small_components(self, mesh):
        """移除三角形数量过少的独立碎片。"""
        if self.min_component_triangles <= 0:
            return mesh

        (
            triangle_clusters,
            cluster_triangle_counts,
            _
        ) = mesh.cluster_connected_triangles()

        triangle_clusters = np.asarray(
            triangle_clusters,
            dtype=np.int64
        )
        cluster_triangle_counts = np.asarray(
            cluster_triangle_counts,
            dtype=np.int64
        )

        remove_mask = (
            cluster_triangle_counts[triangle_clusters]
            < self.min_component_triangles
        )

        removed_count = int(np.count_nonzero(remove_mask))

        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()

        self._log(
            f"已删除小碎片三角面数量: {removed_count}"
        )

        return mesh

    def build_mesh(self, pcd):
        """从已经读取的 Open3D 点云对象生成 Mesh。"""
        pcd = self.prepare_point_cloud(pcd)

        radii_values = self.calculate_bpa_radii(pcd)
        radii = o3d.utility.DoubleVector(radii_values)

        self._log("开始 Ball Pivoting 重建……")

        mesh = (
            o3d.geometry.TriangleMesh
            .create_from_point_cloud_ball_pivoting(
                pcd,
                radii
            )
        )

        if len(mesh.vertices) == 0:
            raise RuntimeError("BPA 没有生成 Mesh 顶点")

        if len(mesh.triangles) == 0:
            raise RuntimeError("BPA 没有生成三角面")

        self._log(
            f"初始 Mesh: {len(mesh.vertices)} 个顶点，"
            f"{len(mesh.triangles)} 个三角面"
        )

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()

        # 删除非流形边可能增加孔洞，因此默认关闭。
        if self.remove_non_manifold_edges:
            mesh.remove_non_manifold_edges()

        mesh.remove_unreferenced_vertices()

        # 移除独立的小碎片
        mesh = self.remove_small_components(mesh)

        # 统一相邻三角面的连接方向
        mesh.orient_triangles()

        self._log("正在转移点云颜色……")
        mesh = self.transfer_colors_to_mesh(mesh, pcd)

        if self.save_normals:
            self._log("正在转移点云法向量……")
            mesh = self.transfer_normals_to_mesh(mesh, pcd)
        else:
            # 只在重建过程中使用法向，不保存到最终 Mesh。
            mesh.vertex_normals = o3d.utility.Vector3dVector()
            mesh.triangle_normals = o3d.utility.Vector3dVector()

        self._log(
            f"最终 Mesh: {len(mesh.vertices)} 个顶点，"
            f"{len(mesh.triangles)} 个三角面"
        )

        return mesh

    def convert(self, pcd, output_path):
        """
        读取点云文件，生成 Mesh 并保存。

        返回：
            Open3D TriangleMesh
        """
        # input_path = Path(input_path)
        # output_path = Path(output_path)

        # if output_path.suffix.lower() != ".ply":
        #     raise ValueError(
        #         "彩色 Mesh 必须保存为 PLY，"
        #         "STL 不能保存标准顶点颜色"
        #     )

        # self._log(f"读取点云: {input_path}")

        # pcd = o3d.io.read_point_cloud(str(input_path))

        mesh = self.build_mesh(pcd)

        # output_path.parent.mkdir(
        #     parents=True,
        #     exist_ok=True
        # )

        ok = o3d.io.write_triangle_mesh(
            str(output_path),
            mesh,
            write_ascii=False,
            write_vertex_normals=self.save_normals,
            write_vertex_colors=True
        )

        if not ok:
            raise RuntimeError(
                f"Mesh 保存失败: {output_path}"
            )

        self._log(f"Mesh 已保存: {output_path}")

        return mesh