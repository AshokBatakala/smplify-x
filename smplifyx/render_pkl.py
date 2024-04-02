# -*- coding: utf-8 -*-

#for Ashok: direction for Usage
# python smplifyx/render_pkl.py -c /path/to/conf.yaml --pkl /path/to/000.pkl 

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
# Contact: Vassilis choutas, vassilis.choutas@tuebingen.mpg.de

import os
import os.path as osp

import argparse
import pickle
import torch
import smplx

from cmd_parser import parse_config
from human_body_prior.tools.model_loader import load_vposer

from utils import JointMapper
import pyrender
import trimesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', nargs='+', type=str, required=True,
                        help='The pkl files that will be read')

    args, remaining = parser.parse_known_args()

    pkl_paths = args.pkl


    # print("Ashok debug 1 \n remaining: ", remaining, type(remaining))
    # args = parse_config(remaining)

    # =============== Ashok wrote this part =================
    # parse_config is throwing error : : Couldn't parse config file: could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'
    # just below code is a temperary fix for this issue


    if remaining[0] == '--config' or remaining[0] == '-c':
        config_file = remaining[1]

        import yaml
        import numpy as np

        # Load the YAML file
        with open(config_file, 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)  # Use FullLoader instead of SafeLoader
    print(f"config file is read from {remaining[1]}")

    #=======================================================
            
    dtype = torch.float32
    use_cuda = args.get('use_cuda', True)
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_type = args.get('model_type', 'smplx')
    print('Model type:', model_type)
    print(args.get('model_folder'))
    model_params = dict(model_path=args.get('model_folder'),
                        #  joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    model = smplx.create(**model_params)
    model = model.to(device=device)

    batch_size = args.get('batch_size', 1)
    use_vposer = args.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    vposer_ckpt = args.get('vposer_ckpt', '')
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        if use_vposer:
            with torch.no_grad():
                pose_embedding[:] = torch.tensor(
                    data['body_pose'], device=device, dtype=dtype)

        est_params = {}
        for key, val in data.items():
            if key == 'body_pose' and use_vposer:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(1, -1)
                if model_type == 'smpl':
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                est_params['body_pose'] = body_pose
            else:
                est_params[key] = torch.tensor(val, dtype=dtype, device=device) # adds all other params of the pkl file

        # ============ change the shape of the body ============
        factor = -2.0
        est_params['betas'] = torch.ones_like(est_params['betas']) * factor

        # =====================================================
        model_output = model(**est_params)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        
        # # ------------- Ashok added this part ----------------
        # import numpy as np
        # import pyrender
        # import trimesh
        # from PIL import Image

        # # Load a texture image
        # tm_path ="/media/Ext_4T_SSD/ASHOK_PART2/default_texture.jpg"
        # texture_image = np.array(Image.open(tm_path))

        # # Create a Texture object
        # texture = pyrender.Texture(source=texture_image, encoding='RGB', wrap_mode='REPEAT')


        # # Create a Material object with the texture
        # material = pyrender.MetallicRoughnessMaterial(baseColorTexture=texture)

        # # Create the mesh with the material
        # # mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)

        # ---------------------------------------------------
        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        pyrender.Viewer(scene, use_raymond_lighting=True)

        # # Ashok added this part ==========================
        # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        # camera_pose = np.eye(4)
        # scene.add(camera, pose=camera_pose)
        # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        # scene.add(light, pose=camera_pose)
        # r = pyrender.OffscreenRenderer(640, 480)
        # color, depth = r.render(scene)
        
        # image_path = "/media/Ext_4T_SSD/ASHOK_PART2/color.png"
        # depth_path = "/media/Ext_4T_SSD/ASHOK_PART2/depth.png"
        # # pyrender.io.write_png(image_path, color)

        # import matplotlib.pyplot as plt
        # plt.imsave(image_path, color)
        # print(f"Image saved to {image_path}")
        # plt.imshow(depth, cmap='gray')
        # plt.savefig(depth_path)
        

