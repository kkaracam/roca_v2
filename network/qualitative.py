import os
import sys

import argparse
import numpy as np
import json
# import open3d as o3d
from PIL import Image
import trimesh
from tqdm import tqdm
from trimesh.exchange.export import export_mesh
from trimesh.util import concatenate as stack_meshes

from roca.engine import Predictor


def main(args):
    BASE_DIR = "/mnt/noraid/karacam/Roca"
    LOG_DIR = os.path.join(BASE_DIR, "roca_log")
    # with open(os.path.join(LOG_DIR, args.model_path, 'frame_aps.json'), 'r') as f:
    with open(os.path.join(LOG_DIR, 'joint_icp_5class', 'frame_aps.json'), 'r') as f:
        frame_aps = json.load(f)
    frame_aps = sorted(frame_aps.items(), key=lambda x: x[1], reverse=True)
    nonzero_aps = [x for x in frame_aps if x[1] > 0]
    zero_aps = [x for x in frame_aps if x[1] == 0.]
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom5Class/scan2cad_instances_val.json', 'r') as f:
        gt_instances = json.load(f)
    model_base = os.path.join(LOG_DIR, args.model_path)
    predictor = Predictor(
        data_dir=(os.path.join(BASE_DIR,args.data_dir)),
        model_path=os.path.join(model_base, 'model_final.pth'),
        config_path=(os.path.join(model_base,'config.yaml')),
        wild=args.wild,
        thresh=0.70,
    )
    to_file = args.output_dir != 'none'

    output_dir = os.path.join(model_base, args.output_dir)
    if to_file and (not os.path.exists(output_dir)):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, 'good'))
        os.mkdir(os.path.join(output_dir, 'bad'))
        os.mkdir(os.path.join(output_dir, 'gt'))
    def process_frame(fname, quality):
        scene = fname.split('/')[0]
        img_name = fname.split('/')[-1]
        #img_fname = "scene0461_00/color/000200.jpg"
        img = Image.open(os.path.join(BASE_DIR, "Data/Images/tasks/scannet_frames_25k", fname))
        img = np.asarray(img)
        instances, cad_ids, params = predictor(img, scene=scene)

        if fname == 'scene0461_00/color/000200.jpg':
            meshes = predictor.output_to_mesh(
                instances,
                cad_ids,
                # Table works poorly in the wild case due to size diversity
                # excluded_classes={'table'} if args.wild else (),
                as_open3d=not to_file,
                params=params,
                fname=fname
            )
            ret_mesh = trimesh.Trimesh(
                vertices=meshes[2].verts_list()[0].numpy(),
                faces=meshes[2].faces_list()[0].numpy()
            )
            def_mesh = meshes[3]
            ret_mesh.export(os.path.join(output_dir, 'ret_mesh.obj'))
            def_mesh.export(os.path.join(output_dir, 'def_mesh.obj'))
            exit()

        meshes = predictor.output_to_mesh(
            instances,
            cad_ids,
            # Table works poorly in the wild case due to size diversity
            # excluded_classes={'table'} if args.wild else (),
            as_open3d=not to_file,
            params=params
        )

        if predictor.can_render:
            rendering, ids = predictor.render_meshes(meshes)
            mask = ids > 0

            overlay = img.copy()
            overlay[mask] = np.clip(
                0.8 * rendering[mask] * 255 + 0.2 * overlay[mask], 0, 255
            ).astype(np.uint8)

            if to_file:
                Image.fromarray(overlay).save(
                    os.path.join(output_dir, quality, f"{scene}_{img_name}")
                )
            else:
                print("Use export!")
                exit()
            
            # gts = []
            # for x in gt_instances['annotations']:
            #     try:
            #         if gt_instances['images'][x['image_id']]['file_name'] == fname:
            #             gts.append(x)
            #     except IndexError as e:
            #         print(e)
            #         print(len(gt_instances['images']), x['image_id'], fname)
            #         exit()
            gt_imgs = {i['id']: i for i in gt_instances['images']}
            gts = [x for x in gt_instances['annotations'] if gt_imgs[x['image_id']]['file_name'].split('/', 2)[-1] == fname]

            gt_meshes = [predictor.annot_meshes(x) for x in gts]
            gt_meshes.append(predictor._camera_mesh)
            gt_rendering, gt_ids = predictor.render_meshes(gt_meshes)
            gt_mask = gt_ids > 0

            overlay = img.copy()
            overlay[gt_mask] = np.clip(
                0.8 * gt_rendering[gt_mask] * 255 + 0.2 * overlay[gt_mask], 0, 255
            ).astype(np.uint8)
            if to_file:
                Image.fromarray(overlay).save(
                    os.path.join(output_dir, 'gt', f"{scene}_{img_name}")
                )
                
        if to_file:
            out_file = os.path.join(output_dir, quality, f"mesh_{scene}_{img_name.split('.')[0]}.ply")
            export_mesh(stack_meshes(meshes), out_file, file_type='ply')
            out_file = os.path.join(output_dir, 'gt', f"gt_mesh_{scene}_{img_name.split('.')[0]}.ply")
            export_mesh(stack_meshes(gt_meshes), out_file, file_type='ply')
        else:
            print("Use export!")
            exit()

    for _fname in tqdm(nonzero_aps[:100]):
        process_frame(_fname[0], 'good')
    # for _fname in tqdm(nonzero_aps[-20:]):
    #     process_frame(_fname[0], 'bad')
    # for _fname in zero_aps[:20]:
    #     scene = _fname.split()[0]
    #     img_name = _fname.split()[-1]
    #     with Image.open(os.path.join(BASE_DIR, "Data/Images/tasks/scannet_frames_25k", _fname)) as img:
    #         img.save(os.path.join(args.output_dir, 'zero', img_id))

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    # parser.add_argument('--config_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--wild', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    main(args)
