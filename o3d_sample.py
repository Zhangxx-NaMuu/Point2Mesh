from typing import Mapping
import open3d as o3d


def sampling_mesh_to_point(mesh):
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    # pcd =mesh.sample_points_uniformly(number_of_points=25000)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud('1_out.ply', pcd, write_ascii=True)
    o3d.io.write_triangle_mesh('1_in.obj', mesh, write_ascii=True)
    
    
if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh('1_test.ply')
    print(mesh)
    sampling_mesh_to_point(mesh)