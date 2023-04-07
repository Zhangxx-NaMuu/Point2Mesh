import torch
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_objs_as_meshes, save_ply
import time

'''
you can use this code to sample ground-truth meshes
input_obj -> mesh to sample
output_ply -> sampled points and normals from input_obj

'''

device = torch.device('cpu')
num_samples = 7000
input_obj = '/zxx/code/point2mesh-master/36_0.obj'
output_ply = '/zxx/code/point2mesh-master/36_0_pc.ply'

mesh = load_objs_as_meshes([input_obj], device=device)
xyz, normals = sample_points_from_meshes(mesh, num_samples=num_samples, return_normals=True)
save_ply(output_ply, verts=xyz[0, :], verts_normals=normals[0, :])
