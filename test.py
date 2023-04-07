import torch
from models.layers.mesh import Mesh, PartMesh
from models.networks import init_net, sample_surface, local_nonuniform_penalty
import utils
import numpy as np
from models.losses import chamfer_distance, BeamGapLoss
from options import Options
import time
import os

options = Options()
opts = options.args
print(opts.load_model)

torch.manual_seed(opts.torch_seed)
device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
# device = torch.device('cpu')
print('device: {}'.format(device))

# initial mesh
mesh = Mesh(opts.initial_mesh, device=device, hold_history=True)

# input point cloud
input_xyz, input_normals = utils.read_pts(opts.input_pc)
# normalize point cloud based on initial mesh
input_xyz /= mesh.scale
input_xyz += mesh.translations[None, :]
input_xyz = torch.Tensor(input_xyz).type(options.dtype()).to(device)[None, :, :]
input_normals = torch.Tensor(input_normals).type(options.dtype()).to(device)[None, :, :]

part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
print(f'number of parts {part_mesh.n_submeshes}')
net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)
# print(net)



mesh = part_mesh.main_mesh
num_faces = int(np.clip(len(mesh.faces) * 1.5, len(mesh.faces), opts.max_faces))
print(num_faces)
print(len(mesh.faces))
# up-sample mesh
mesh = utils.manifold_upsample(mesh, opts.save_path, Mesh,
                                num_faces=min(num_faces, opts.max_faces),
                                res=opts.manifold_res, simplify=True)

part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
    
with torch.no_grad():
    # # 保存最新模型
    # torch.save(net, os.path.join(opts.save_path, 'latest_model.pth'))
    # 保存最新网格
    mesh.export(os.path.join(opts.save_path, 'new_recon.obj'))
