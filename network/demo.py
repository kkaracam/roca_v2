import os
import sys

import argparse
import numpy as np
# import open3d as o3d
from PIL import Image
from trimesh.exchange.export import export_mesh
from trimesh.util import concatenate as stack_meshes

from roca.engine import Predictor


def main(args):
    predictor = Predictor(
        data_dir=(os.path.join("/mnt/noraid/karacam/Roca/roca_log",args.data_dir)),
        model_path=(os.path.join("/mnt/noraid/karacam/Roca/roca_log",args.model_path)),
        config_path=(os.path.join("/mnt/noraid/karacam/Roca/roca_log",args.config_path)),
        wild=args.wild,
    )
    to_file = args.output_dir != 'none'

    for name, scene in zip(
        ('3m', 'sofa', 'lab', 'desk'),
        ('scene0474_02', 'scene0207_00', 'scene0378_02', 'scene0474_02')
    ):
        # name = '900'
        # scene = 'scene0474_03'
        img = Image.open(os.path.join('assets', '{}.jpg'.format(name)))#.resize((640,480))
        # img = Image.open(os.path.join('/mnt/noraid/karacam/AlignmentNew/Resized400k/tasks/scannet_frames_25k/scene0474_03/color/', '{}.jpg'.format(name)))
        img = np.asarray(img)
        instances, cad_ids, params, joint_idx = predictor(img, scene=scene)
        # print(instances)
        # exit()
        meshes = predictor.output_to_mesh(
            instances,
            cad_ids,
            # Table works poorly in the wild case due to size diversity
            excluded_classes={'table'} if args.wild else (),
            as_open3d=not to_file,
            params=params,
            joint_idx=joint_idx
        )

        if to_file:
            os.makedirs(args.output_dir, exist_ok=True)

        if predictor.can_render:
            rendering, ids = predictor.render_meshes(meshes)
            mask = ids > 0
            # print(rendering.sum())
            # exit()
            overlay = img.copy()
            overlay[mask] = np.clip(
                0.8 * rendering[mask] * 255 + 0.2 * overlay[mask], 0, 255
            ).astype(np.uint8)
            # for i in range(instances.pred_masks.shape[0]):
            #     overlay[instances.pred_masks[i]] = [255,255,255]

            if to_file:
                Image.fromarray(overlay).save(
                    os.path.join(args.output_dir, 'overlay_{}.jpg'.format(name))
                )
            else:
                print("Use export!")
                exit()
                # img = o3d.geometry.Image(overlay)
                # o3d.visualization.draw_geometries([img], height=480, width=640)

        if to_file:
            out_file = os.path.join(args.output_dir, 'mesh_{}.ply'.format(name))
            export_mesh(stack_meshes(meshes), out_file, file_type='ply')
        else:
            print("Use export!")
            exit()
            # o3d.visualization.draw_geometries(meshes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--wild', action='store_true')
    parser.add_argument('--output_dir', default='none')
    args = parser.parse_args(sys.argv[1:])
    main(args)
