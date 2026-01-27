import numpy as np
import open3d as o3d
#
# bunny = o3d.data.BunnyMesh()
# mesh = o3d.io.read_triangle_mesh(bunny.path)
# mesh.compute_vertex_normals()
#
# pcd = mesh.sample_points_poisson_disk(750)
# o3d.visualization.draw_geometries([pcd])
# alpha = 0.010
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)






print('create noisy mesh')
knot_mesh = o3d.data.KnotMesh()
mesh_in = o3d.io.read_triangle_mesh(knot_mesh.path)
vertices = np.asarray(mesh_in.vertices)
noise = 5
vertices += np.random.uniform(0, noise, size=vertices.shape)
mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
mesh_in.compute_vertex_normals()

print('filter with average with 1 iteration')
# mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
# mesh_out.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_out])


print('filter with Laplacian with 50 iterations')
mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=50)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])