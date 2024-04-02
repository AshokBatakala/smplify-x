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
    parser.add_argument('--vis_values_pkl', type=str, required=True,)

    args, remaining = parser.parse_known_args()

    pkl_paths = args.pkl
    vis_values_pkl = args.vis_values_pkl


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
        # factor = -2.0
        # est_params['betas'] = torch.ones_like(est_params['betas']) * factor

        # =====================================================
        model_output = model(**est_params)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        
        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        # pyrender.Viewer(scene, use_raymond_lighting=True)
        
        import pickle

        def render_mesh(vis_values_pkl, mesh ):
            #load the values from the pickle file
            with open(vis_values_pkl, 'rb') as vis_file:
                vis_vals = pickle.load(vis_file)
            #create a pyrender scene
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
            #create a pyrender mesh
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(1.0, 1.0, 0.9, 1.0))
            mesh = pyrender.Mesh.from_trimesh(
                mesh,
                material=material)
            scene.add(mesh, 'mesh')
            #create a pyrender camera
            camera = pyrender.camera.IntrinsicsCamera(
                fx=vis_vals['focal_length'], fy=vis_vals['focal_length'],
                cx=vis_vals['camera_center'][0], cy=vis_vals['camera_center'][1])
            scene.add(camera, pose=vis_vals['camera_pose'])
            #create a pyrender light
            light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
            scene.add(light, pose=vis_vals['camera_pose'])
            #create a pyrender renderer
            r = pyrender.OffscreenRenderer(vis_vals['W'], vis_vals['H'])
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
            return color
        # invert this transform
            #     rot = trimesh.transformations.rotation_matrix(
            # np.radians(180), [1, 0, 0])
        
        import numpy as np
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        
        out_mesh.apply_transform(rot)

        # render the mesh
        color = render_mesh(vis_values_pkl, out_mesh)
        import matplotlib.pyplot as plt
        plt.imshow(color)
        plt.show()
        plt.imsave('rendered.png', color)


