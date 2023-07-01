from typing import Any, List, Iterable, Tuple, Union

import numpy as np
import torch
import trimesh
import json
import pickle
import os

from detectron2.modeling import build_model
from detectron2.structures import Instances

from roca.config import roca_config
from roca.data import CADCatalog
from roca.data.constants import CAD_TAXONOMY, COLOR_BY_CLASS
from roca.data.datasets import register_scan2cad
from roca.structures import Intrinsics
from roca.utils.alignment_errors import translation_diff
from roca.utils.linalg import make_M_from_tqs
from roca.modeling.retrieval_head.joint_retrieve_deform_ops import get_model, get_source_info_mesh, get_shape_numpy

try:
    import sys
    sys.path.insert(-1, '../renderer')
    from scan2cad_rasterizer import Rasterizer
except ImportError:
    Rasterizer = None


class Predictor:
    def __init__(
        self,
        data_dir: str,
        model_path: str,
        config_path: str,
        thresh: float = 0.5,
        wild: bool = False
    ):
        cfg = roca_config('Scan2CAD', 'Scan2CAD')
        cfg.merge_from_file(config_path)
        if wild:
            cfg.MODEL.WILD_RETRIEVAL_ON = True
        cfg.MODEL.INSTANCES_CONFIDENCE_THRESH = thresh

        model = build_model(cfg)
        backup = torch.load(model_path)
        model.load_state_dict(backup['model'])

        model.eval()
        model.requires_grad_(False)

        data_name = 'Scan2CADVal'
        register_scan2cad(data_name, {}, '', data_dir, '', '', 'val')

        cad_manager = CADCatalog.get(data_name)
        self.is_voxel = cfg.INPUT.CAD_TYPE == 'voxel'
        self.joint = cfg.MODEL.RETRIEVAL_MODE == 'joint'
        points, ids = cad_manager.batched_points_and_ids(volumes=self.is_voxel)
        model.set_cad_models(points, ids, cad_manager.scene_alignments)
        model.embed_cad_models()

        self.wild = wild
        if wild:
            data_name = data_name.replace('Val', 'Train')
            register_scan2cad(data_name, {}, '', data_dir, '', '', 'train')
            self.train_cad_manager = CADCatalog.get(data_name)
            train_points, train_ids = self.train_cad_manager.batched_points_and_ids(
                volumes=self.is_voxel
            )
            model.set_train_cads(train_points, train_ids)
            model.embed_train_cads()
        else:
            data_name = data_name.replace('Val', 'Train')
            register_scan2cad(data_name, {}, '', data_dir, '', '', 'train')
            self.train_cad_manager = CADCatalog.get(data_name)
        self.model = model
        self.cad_manager = cad_manager

        with open('./assets/camera.obj') as f:
            cam = trimesh.load(f, file_type='obj', force='mesh')
        cam.apply_scale(0.25)
        cam.visual.face_colors = [100, 100, 100, 255]
        self._camera_mesh = cam

        self.scene_rot = np.array([
            [1, 0, 0, 0],
            [0, np.cos(np.pi), -np.sin(np.pi), 0],
            [0, np.sin(np.pi), np.cos(np.pi), 0],
            [0, 0, 0, 1]
        ])

        if self.joint:
            with open("/home/karacam/Thesis/ROCA/shape2part_ext.json", 'r') as f:
                shape2part = json.load(f)
            part2shape = dict([(value, key) for k in shape2part for key, value in shape2part[k]['model_names'].items()])
            self.source_info = {}
            for c_name, _num_s in zip(['chair','table','storagefurniture','display','bin'], [522,676,306,138,79]):
                filename_pickle = cfg.JOINT_SRC_PKL.format(c_name, _num_s)
                with open(filename_pickle, 'rb') as handle:
                    sources = pickle.load(handle)['sources']
                self.src_data_fol = cfg.JOINT_SRC_DIR.format(c_name)
                print("Loading source mesh info...")
                for src in sources:
                    src_filename = str(src) + "_leaves.h5"
                    shapeid = part2shape[str(src)]
                    self.source_info[shapeid] = (get_model(os.path.join(self.src_data_fol, src_filename), pred=True))
            print("Done.")
                # filename_pickle = cfg.JOINT_SRC_PKL
                # with open(filename_pickle, 'rb') as handle:
                #     sources = pickle.load(handle)['sources']
                # self.src_data_fol = cfg.JOINT_SRC_DIR
                # print("Loading source mesh info...")
                # for src in sources:
                #     src_filename = str(src) + "_leaves.h5"
                #     shapeid = part2shape[str(src)]
                #     self.source_info[shapeid] = (get_model(os.path.join(self.src_data_fol, src_filename), pred=True))
                # print("Done.")

    @property
    def can_render(self):
        return Rasterizer is not None

    @torch.no_grad()
    def __call__(
        self,
        image_rgb: np.ndarray,
        f: Union[np.ndarray, float] = 435.,
        scene: str = 'scene0474_02'
    ) -> Tuple[Instances, List[Tuple[str, str]]]:

        inputs = {'scene': scene}
        inputs['image'] = torch.as_tensor(
            np.ascontiguousarray(image_rgb[:, :, ::-1].transpose(2, 0, 1))
        )
        if isinstance(f, np.ndarray):
            inputs['intrinsics'] = f[:3, :3]
        else:
            inputs['intrinsics'] = Intrinsics(torch.tensor([
                [f, 0., image_rgb.shape[1] / 2],
                [0., f, image_rgb.shape[0] / 2],
                [0., 0., 1.]
            ]))
        outputs = self.model([inputs])[0]
        if len(outputs["instances"]) == 0:
            return outputs['instances'].to('cpu'), []
        cad_ids = outputs['wild_cad_ids'] if self.wild else outputs['cad_ids']
        params = outputs['pred_params'] if ('pred_params' in outputs) and self.joint else None
        return outputs['instances'].to('cpu'), cad_ids, params#.to('cpu')

    def output_to_mesh(
        self,
        instances: Instances,
        cad_ids: List[Tuple[str, str]],
        min_dist_3d: float = 0.4,
        excluded_classes: Iterable[str] = (),
        nms_3d: bool = True,
        as_open3d: bool = False,
        params=None,
        fname=None
    ) -> Union[List[trimesh.Trimesh], List[Any]]:

        meshes = []
        trans_cls_scores = []
        if fname is not None:
            for i in range(len(instances)):
                cad_id = cad_ids[i]
                if cad_id is None:
                    continue
                ret_mesh = self.train_cad_manager.model_by_id(*cad_id)
                pred_params = params[i].cpu().numpy()
                default_param, vertices_mat, faces, constraint_proj_mat = self.source_info[cad_id[1]]
                curr_param = np.expand_dims(pred_params, -1)
                curr_mat, curr_default_param, curr_conn_mat = get_source_info_mesh(vertices_mat, default_param, constraint_proj_mat, curr_param.shape[0])
                output_vertices = get_shape_numpy(curr_mat, curr_param, curr_default_param.T, connectivity_mat=curr_conn_mat)
                def_mesh = trimesh.Trimesh(
                    vertices=output_vertices,
                    faces=faces
                )
                meshes.append(ret_mesh)
                meshes.append(def_mesh)
            return meshes
        for i in range(len(instances)):
            cad_id = cad_ids[i]
            if cad_id is None:
                continue
            if CAD_TAXONOMY[int(cad_id[0])] in excluded_classes:
                continue

            trans_cls_scores.append((
                instances.pred_translations[i],
                instances.pred_classes[i].item(),
                instances.scores[i].item(),
            ))

            if params is not None:
                pred_params = params[i].cpu().numpy()
                default_param, vertices_mat, faces, constraint_proj_mat = self.source_info[cad_id[1]]
                curr_param = np.expand_dims(pred_params, -1)
                curr_mat, curr_default_param, curr_conn_mat = get_source_info_mesh(vertices_mat, default_param, constraint_proj_mat, curr_param.shape[0])
                output_vertices = get_shape_numpy(curr_mat, curr_param, curr_default_param.T, connectivity_mat=curr_conn_mat)
                mesh = trimesh.Trimesh(
                    vertices=output_vertices,
                    faces=faces
                )
            else:
                if self.wild or self.joint:
                    mesh = self.train_cad_manager.model_by_id(*cad_id)
                else:
                    mesh = self.cad_manager.model_by_id(*cad_id)
                mesh = trimesh.Trimesh(
                    vertices=mesh.verts_list()[0].numpy(),
                    faces=mesh.faces_list()[0].numpy()
                )

            trs = make_M_from_tqs(
                instances.pred_translations[i].tolist(),
                instances.pred_rotations[i].tolist(),
                instances.pred_scales[i].tolist()
            )
            mesh.apply_transform(self.scene_rot @ trs)

            color = COLOR_BY_CLASS[int(cad_ids[i][0])]
            if as_open3d:
                mesh = mesh.as_open3d
                mesh.paint_uniform_color(color)
                mesh.compute_vertex_normals()
            else:
                mesh.visual.face_colors = [*(255 * color), 255]
            meshes.append(mesh)

        if nms_3d:
            keeps = self._3d_nms(trans_cls_scores, min_dist_3d)
            meshes = [m for m, b in zip(meshes, keeps) if b]

        if as_open3d:
            cam = self._camera_mesh
            cam = cam.as_open3d
            cam.compute_vertex_normals()
            meshes.append(cam)
        else:
            meshes.append(self._camera_mesh)

        return meshes

    def render_meshes(
        self, meshes: Union[List[trimesh.Trimesh], List[Any]], f: float = 435.,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if Rasterizer is None:
            raise ImportError('scan2cad-rasterizer not installed')

        inv_rot = np.linalg.inv(self.scene_rot)
        raster = Rasterizer(480, 360, f, f, 240., 180, False, True)
        colors = {}
        # print(meshes)
        # exit()
        for i, mesh in enumerate(meshes[:-1], start=1):
            # print(isinstance(mesh, trimesh.Trimesh))
            if isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.copy()
                mesh.apply_transform(inv_rot)
                raster.add_model(
                    np.asarray(mesh.faces, dtype=raster.index_dtype),
                    np.asarray(mesh.vertices, dtype=raster.scalar_dtype),
                    i,
                    np.asarray(mesh.face_normals, raster.scalar_dtype)
                )
                colors[i] = np.asarray(mesh.visual.face_colors)[0][:3] / 255
            else:
                # Open3D
                mesh = type(mesh)(mesh)
                mesh.compute_triangle_normals()
                mesh.transform(inv_rot)
                raster.add_model(
                    np.asarray(mesh.triangles, dtype=raster.index_dtype),
                    np.asarray(mesh.vertices, dtype=raster.scalar_dtype),
                    i,
                    np.asarray(mesh.triangle_normals, raster.scalar_dtype)
                )
                colors[i] = np.asarray(mesh.vertex_colors[0])[:3]

        raster.rasterize()
        raster.set_colors(colors)
        raster.render_colors(0.2)
        return raster.read_color(), raster.read_idx()
    
    def annot_meshes(self, annot):
        mesh = self.cad_manager.model_by_id(annot['model']['catid_cad'], annot['model']['id_cad'])
        mesh = trimesh.Trimesh(
            vertices=mesh.verts_list()[0].numpy(),
            faces=mesh.faces_list()[0].numpy()
        )
        trs = make_M_from_tqs(
            annot['t'],
            annot['q'],
            annot['s']
        )
        mesh.apply_transform(self.scene_rot @ trs)
        color = COLOR_BY_CLASS[int(annot['model']['catid_cad'])]
        mesh.visual.face_colors = [*(255 * color), 255]
        return mesh

    @staticmethod
    def _3d_nms(tcs, min_dist):
        keeps = [True for _ in tcs]
        if min_dist <= 0:
            return keeps
        for i, (t, c, s) in enumerate(tcs):
            if any(
                c_ == c and s_ > s and translation_diff(t_, t) < min_dist
                for t_, c_, s_ in tcs
            ):
                keeps[i] = False
        return keeps

