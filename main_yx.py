import torch
from models.layers.mesh import Mesh, PartMesh
from models.networks import init_net, sample_surface, local_nonuniform_penalty, get_scheduler,populate_e
import utils
import numpy as np
from models.losses import chamfer_distance, BeamGapLoss
from options import Options
import time
import os

def save_checkpoint(checkpoint_path, net, optimizer, rand_verts, scheduler,beamgap_loss):
    print("====================================================")
    print(optimizer.state_dict())
    print("====================================================")
    print(beamgap_loss)
    print("====================================================")
    state_dict={}
    state_dict['net'], state_dict['optimizer']= net.state_dict(), optimizer.state_dict()
    state_dict['beamgap_loss']=beamgap_loss
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(state_dict, checkpoint_path)




def train():
    #加载配置
    options = Options()
    opts = options.args
    torch.manual_seed(opts.torch_seed)
    device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
    # 初始化网格
    mesh = Mesh(opts.initial_mesh, device=device, hold_history=True)
    # 输入点云
    input_xyz, input_normals = utils.read_pts(opts.input_pc)
    # 归一化点云
    input_xyz /= mesh.scale
    input_xyz += mesh.translations[None, :]
    input_xyz = torch.Tensor(input_xyz).type(options.dtype()).to(device)[None, :, :]
    input_normals = torch.Tensor(input_normals).type(options.dtype()).to(device)[None, :, :]

    #初始化网络
    part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
    print(f'number of parts {part_mesh.n_submeshes}')
    net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)
    beamgap_loss = BeamGapLoss(device)


    #开始迭代
    if opts.beamgap_iterations > 0:
        print('beamgap on')
        beamgap_loss.update_pm(part_mesh, torch.cat([input_xyz, input_normals], dim=-1))

    for i in range(opts.iterations):
        num_samples = options.get_num_samples(i % opts.upsamp)
        if opts.global_step:
            optimizer.zero_grad()
        start_time = time.time()
        for part_i, est_verts in enumerate(net(rand_verts, part_mesh)):
            if not opts.global_step:
                optimizer.zero_grad()
            part_mesh.update_verts(est_verts[0], part_i)
            num_samples = options.get_num_samples(i % opts.upsamp)
            recon_xyz, recon_normals = sample_surface(part_mesh.main_mesh.faces, part_mesh.main_mesh.vs.unsqueeze(0),
                                                    num_samples)
            # calc chamfer loss w/ normals
            recon_xyz, recon_normals = recon_xyz.type(options.dtype()), recon_normals.type(options.dtype())
            xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, input_xyz, x_normals=recon_normals,
                                                                    y_normals=input_normals,
                                                                    unoriented=opts.unoriented)

            if (i < opts.beamgap_iterations) and (i % opts.beamgap_modulo == 0):
                loss = beamgap_loss(part_mesh, part_i)
            else:
                loss = (xyz_chamfer_loss + (opts.ang_wt * normals_chamfer_loss))
            if opts.local_non_uniform > 0:
                loss += opts.local_non_uniform * local_nonuniform_penalty(part_mesh.main_mesh).float()
            loss.backward()
            if not opts.global_step:
                optimizer.step()
                scheduler.step()
            part_mesh.main_mesh.vs.detach_()
        if opts.global_step:
            optimizer.step()
            scheduler.step()
        end_time = time.time()

        if i % 1 == 0:
            print(f'{os.path.basename(opts.input_pc)}; iter: {i} out of: {opts.iterations}; loss: {loss.item():.4f};'
                f' sample count: {num_samples}; time: {end_time - start_time:.2f}')
        if i % opts.export_interval == 0 and i > 0:
            print('exporting reconstruction... current LR: {}'.format(optimizer.param_groups[0]['lr']))
            with torch.no_grad():
                part_mesh.export(os.path.join(opts.save_path, f'recon_iter_{i}.obj'))
                    # 保存最新模型
                save_checkpoint("./last_model.pt", net, optimizer, rand_verts, scheduler,beamgap_loss)
                torch.save(net,"./last_model_torch.pt")


def test(checkpoint_path):
     #加载配置
    options = Options()
    opts = options.args
    torch.manual_seed(opts.torch_seed)
    device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
    # 初始化网格
    mesh = Mesh(opts.initial_mesh, device=device, hold_history=True)
    # 输入点云
    input_xyz, input_normals = utils.read_pts(opts.input_pc)
    # 归一化点云
    input_xyz /= mesh.scale
    input_xyz += mesh.translations[None, :]
    input_xyz = torch.Tensor(input_xyz).type(options.dtype()).to(device)[None, :, :]
    input_normals = torch.Tensor(input_normals).type(options.dtype()).to(device)[None, :, :]

    #加载网络
    part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
    print(f'number of parts {part_mesh.n_submeshes}')

    net,_, _, _ = init_net(mesh, part_mesh, device, opts)
    net=torch.load(checkpoint_path)
    #net.eval()
    rand_verts = populate_e([mesh])
    with torch.no_grad():
        for part_i, est_verts in enumerate(net(rand_verts, part_mesh)):
            part_mesh.update_verts(est_verts[0], part_i)
            part_mesh.main_mesh.vs.detach_()
        part_mesh.export(os.path.join(opts.save_path, f'yx.obj'))



#test("last_model.pt")
test("last_model_torch.pt")
#train()



