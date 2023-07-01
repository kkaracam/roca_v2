import json
import os
import pickle as pkl
from collections import Counter, defaultdict, OrderedDict
from copy import deepcopy
from itertools import product
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    OrderedDict as OrderedDictType,
    Union,
)

import numpy as np
import trimesh
from trimesh.voxel.creation import local_voxelize
import quaternion  # noqa: F401
import torch
from pandas import DataFrame, read_csv
from tabulate import tabulate
import h5py
from tqdm import tqdm
from roca.data import CADCatalog

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Instances

from roca.data import CategoryCatalog
from roca.data.constants import (
    CAD_TAXONOMY,
    CAD_TAXONOMY_REVERSE,
    IMAGE_SIZE,
)
from roca.structures import Rotations
from roca.utils.alignment_errors import (
    rotation_diff,
    scale_ratio,
    translation_diff,
)
from roca.utils.linalg import decompose_mat4, make_M_from_tqs, transform_mesh
from roca.utils.ap import compare_meshes
from roca.modeling.retrieval_head.joint_retrieve_deform_ops import get_model, get_source_info_mesh, get_shape_numpy

from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
# from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.renderer import(
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)

from torch_cluster import fps

NMS_TRANS = 0.4
NMS_ROT = 60
NMS_SCALE = 0.6

TRANS_THRESH = 0.3
ROT_THRESH = 20
SCALE_THRESH = 0.2
VOXEL_IOU_THRESH = 0.5


class Vid2CADEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name: str,
        full_annot: Union[str, List[Dict[str, Any]]],
        cfg=None,
        output_dir: str = '',
        mocking: bool = False,
        exclude: Optional[Iterable[str]]=None,
        grid_file: Optional[str] = None,
        exact_ret: bool = False,
        key_prefix: str = '',
        info_file: str = '',
        src_info: Dict = None
    ):
        self._dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(self._dataset_name)
        self._category_manager = CategoryCatalog.get(self._dataset_name)
        self.cad_manager = CADCatalog.get(dataset_name)
        self.cad_manager_train = CADCatalog.get(cfg.DATASETS.TRAIN[0])

        self.mocking = mocking
        self._output_dir = output_dir

        self._exclude = exclude

        self.DEFORM = True
        CAD_TAXONOMY[90000000] = 'storagefurniture'
        CAD_TAXONOMY_REVERSE['storagefurniture'] = 90000000

        # Parse raw data
        if isinstance(full_annot, list):
            annots = full_annot
        else:
            with open(full_annot) as f:
                annots = json.load(f)
        self._full_annots = annots

        scene_alignments = {}
        scene_counts = defaultdict(lambda: Counter())
        for annot in annots:
            scene = annot['id_scan']
            trs = annot['trs']
            to_scene = np.linalg.inv(make_M_from_tqs(
                trs['translation'], trs['rotation'], trs['scale']
            ))
            alignment = []
            for model in annot['aligned_models']:
                model_catid_cad = int(model['catid_cad'])
                if CAD_TAXONOMY[int(model['catid_cad'])] == 'bookcase' or CAD_TAXONOMY[int(model['catid_cad'])] == 'cabinet':
                    model_catid_cad = CAD_TAXONOMY_REVERSE['storagefurniture']
                if model_catid_cad not in CAD_TAXONOMY:
                    continue
                scene_counts[scene][model_catid_cad] += 1
                mtrs = model['trs']
                to_s2c = make_M_from_tqs(
                    mtrs['translation'],
                    mtrs['rotation'],
                    mtrs['scale']
                )
                t, q, s = decompose_mat4(to_scene @ to_s2c)
                alignment.append({
                    't': t.tolist(),
                    'q': q.tolist(),
                    's': s.tolist(),
                    'catid_cad': model['catid_cad'],
                    'id_cad': model['id_cad'],
                    'sym': model['sym']
                })
            scene_alignments[scene] = alignment

        self._scene_alignments = scene_alignments
        self._scene_counts = scene_counts

        with open("/mnt/noraid/karacam/ThesisData/data/roca_storagefurniture.json", 'r') as f:
            storagefurniture_ids = json.load(f)
            self.storagefurniture_ids = {e[1]:e[0] for e in storagefurniture_ids}

        self.with_grids = grid_file is not None
        self.grid_data = None
        if self.with_grids:
            with open(grid_file, 'rb') as f:
                self.grid_data = pkl.load(f)
        with open(self._metadata.train_grid_file, 'rb') as f:
            self.train_grid_data = pkl.load(f)
        with open(self._metadata.val_grid_file, 'rb') as f:
            self.val_grid_data = pkl.load(f)
        with open(self._metadata.point_file, 'rb') as f:
            point_data = pkl.load(f)
        self.point_data = {(p['catid_cad'], p['id_cad']): p['points'] for p in point_data}

        self.exact_ret = exact_ret
        self.key_prefix = key_prefix
        self.info_file = info_file

        with open("/home/karacam/Thesis/ROCA/shape2part_ext.json", 'r') as f:
            shape2part = json.load(f)
        part2shape = dict([(value, key) for k in shape2part for key, value in shape2part[k]['model_names'].items()])
        self.source_info = {}
        self._metas = {}
        with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom3Class/train_grids_32.pkl", 'rb') as f:
            self.train_grids = pkl.load(f)
        if self.DEFORM:
            if not src_info:
                for i, _num_s in zip([0,1,2,3,4], [522,676,306,138,79]):
                    c_name = self._category_manager.get_name(i)
                    filename_pickle = cfg.JOINT_SRC_PKL.format(c_name, _num_s)
                    with open(filename_pickle, 'rb') as handle:
                        sources = pkl.load(handle)['sources']
                    self.src_data_fol = cfg.JOINT_SRC_DIR.format(c_name)
                    print("Loading source mesh info...")
                    for src in sources:
                        src_filename = str(src) + "_leaves.h5"
                        shapeid = part2shape[str(src)]
                        self.source_info[shapeid] = (get_model(os.path.join(self.src_data_fol, src_filename), pred=True))
                        # with open(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{shapeid}/models/model_normalized.json", 'r') as jf:
                        #     _meta = json.load(jf)
                        # self._metas[shapeid] = _meta
            else:
                self.source_info = src_info
            print("Done.")
        self.noc_tar_trip = []

    def reset(self):
        self.results = defaultdict(list)
        self.poses = defaultdict(list)
        self.object_ids = defaultdict(list)
        self.object_params = defaultdict(list)
        self.object_comps = defaultdict(list)
        self.info_data = defaultdict(list)

    def process(
        self,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]]
    ):
        for input, output in zip(inputs, outputs):
            file_name = input['file_name']
            scene_name = input['file_name'].split('/')[-3]

            if 'instances' not in output:
                continue
            instances = output['instances']
            instances = instances[instances.scores > 0.5]
            has_alignment = instances.has_alignment
            if not has_alignment.any():
                continue
            instances = instances[has_alignment]
            instances = deepcopy(instances)  # avoid modification!
            if instances.has('pred_meshes'):
                instances.remove('pred_meshes')
            self.results[scene_name].append(instances.to('cpu'))

            if 'wild_cad_ids' in output:
                object_ids = output['wild_cad_ids']
                object_ids = [
                    object_ids[i]
                    for i in instances.pred_indices.tolist()
                ]
                self.object_ids[scene_name].append(object_ids)
            elif 'cad_ids' in output:
                object_ids = output['cad_ids']
                object_ids = [
                    object_ids[i]
                    for i in instances.pred_indices.tolist()
                ]
                self.object_ids[scene_name].append(object_ids)
            if 'pred_params' in output:
                object_params = [
                    output['pred_params'][i].cpu().numpy()
                    for i in instances.pred_indices.tolist()
                ]
                self.object_params[scene_name].append(object_params)
            if 'nocs_comp' in output:
                object_comps = [
                    output['nocs_comp'][i].cpu().numpy()
                    for i in instances.pred_indices.tolist()
                ]
                self.object_comps[scene_name].append(object_comps)

            pose_file = file_name\
                .replace('color', 'pose')\
                .replace('.jpg', '.txt')
            with open(pose_file) as f:
                pose_mat = torch.tensor([
                    [float(v) for v in line.strip().split()]
                    for line in f
                ])
            pose_mat = pose_mat.unsqueeze(0).expand(len(instances), 4, 4)
            self.poses[scene_name].append(pose_mat)
        '''if len(self.poses) == 20:
            self.evaluate()

            exit(0)'''

    def process_mock(
        self,
        scene_name: str,
        instances: Instances,
        object_ids=None,
        object_params=None
    ):
        self.results[scene_name] = instances
        self.object_ids[scene_name] = object_ids
        self.object_params[scene_name] = object_params

    def evaluate(self) -> OrderedDictType[str, Dict[str, float]]:
        self._collect_results()
        self._transform_results_to_world_space()
        path = self._write_raw_results()
        self._nms_results()
        self._apply_constraints()
        return self._compute_metrics()
        # return eval_csv(
        #     self._dataset_name,
        #     path,
        #     self._full_annots,
        #     exact_ret=self.exact_ret,
        #     prefix=self.key_prefix,
        #     info_file=self.info_file
        # )

    def evaluate_mock(self) -> OrderedDictType[str, Dict[str, float]]:
        self._nms_results()
        self._apply_constraints()
        return self._compute_metrics()

    def _collect_results(self):
        print('INFO: Collecting results...', flush=True)

        for k, v in self.results.items():
            instances = Instances.cat(v)
            indices = instances.scores.argsort(descending=True)
            self.results[k] = instances[indices]
            self.poses[k] = torch.cat(self.poses[k], dim=0)[indices]

            # NOTE: Objects corresponds to instances,
            # so sort them similar to results
            if k in self.object_ids:
                object_ids = []
                for ids in self.object_ids[k]:
                    object_ids.extend(ids)
                self.object_ids[k] = [object_ids[i] for i in indices.tolist()]
            if k in self.object_params:
                object_params = []
                for params in self.object_params[k]:
                    object_params.extend(params)
                self.object_params[k] = [object_params[i] for i in indices.tolist()]
            if k in self.object_comps:
                object_comps = []
                for comps in self.object_comps[k]:
                    object_comps.extend(comps)
                self.object_comps[k] = [object_comps[i] for i in indices.tolist()]

    def _transform_results_to_world_space(self):
        print('INFO: Transforming results to world space...', flush=True)

        for scene, instances in self.results.items():
            poses = self.poses[scene]

            # TODO: This can be batched
            for i, (pose, t, q, s) in enumerate(zip(
                poses.unbind(0),
                instances.pred_translations.unbind(0),
                instances.pred_rotations.unbind(0),
                instances.pred_scales.unbind(0)
            )):
                pose = pose.numpy().reshape(4, 4)
                mat = make_M_from_tqs(t.tolist(), q.tolist(), s.tolist())
                new_t, new_q, new_s = decompose_mat4(pose @ mat)

                instances.pred_translations[i] = torch.from_numpy(new_t)
                instances.pred_rotations[i] = torch.from_numpy(new_q)
                instances.pred_scales[i] = torch.from_numpy(new_s)

            self.results[scene] = instances
            self.poses[scene] = poses

    def _write_raw_results(self):
        output_dir = self._output_dir
        output_path = os.path.join(output_dir, 'raw_results.csv')
        print(
            'INFO: Writing raw results to {}...'.format(output_path),
            flush=True
        )

        data = defaultdict(lambda: [])
        results = sorted(self.results.items(), key=lambda x: x[0])
        for scene, instances in results:
            data['id_scan'].extend((scene,) * len(instances))
            for c in instances.pred_classes.tolist():
                c_name = self._category_manager.get_name(c)
                cid = CAD_TAXONOMY_REVERSE[c_name]
                # cid = CAD_TAXONOMY_REVERSE[c_name] if not c_name == 'storagefurniture' else c
                data['objectCategory'].append(cid)

            data['alignedModelId'].extend(
                id_cad for _, id_cad in self.object_ids[scene]
            )

            data['tx'].extend(instances.pred_translations[:, 0].tolist())
            data['ty'].extend(instances.pred_translations[:, 1].tolist())
            data['tz'].extend(instances.pred_translations[:, 2].tolist())

            data['qw'].extend(instances.pred_rotations[:, 0].tolist())
            data['qx'].extend(instances.pred_rotations[:, 1].tolist())
            data['qy'].extend(instances.pred_rotations[:, 2].tolist())
            data['qz'].extend(instances.pred_rotations[:, 3].tolist())

            data['sx'].extend(instances.pred_scales[:, 0].tolist())
            data['sy'].extend(instances.pred_scales[:, 1].tolist())
            data['sz'].extend(instances.pred_scales[:, 2].tolist())

            data['object_score'].extend(instances.scores.tolist())

        frame = DataFrame(data=data)
        frame.to_csv(output_path, index=False)
        return output_path

    def _nms_results(self):
        print('INFO: Removing duplicate results...', flush=True)
        for scene, instances in self.results.items():
            pred_trans = instances.pred_translations
            pred_rot = Rotations(instances.pred_rotations).as_quaternions()
            pred_scale = instances.pred_scales
            pred_classes = instances.pred_classes
            num_instances = len(instances)
            # scores = instances.scores

            all_pairs = product(
                range(num_instances), reversed(range(num_instances))
            )
            valid_map = torch.ones(len(instances), dtype=torch.bool)
            for i, j in all_pairs:
                '''if scores[j] >= scores[i]:
                    continue'''
                # NOTE: Assume sorted by score
                if i >= j:
                    continue
                # NOTE: if the high score one was removed earlier,
                # do not drop the low score one
                if not valid_map[i] or not valid_map[j]:
                    continue
                if pred_classes[j] != pred_classes[i]:
                    continue

                # object_ids = self.object_ids[scene]
                # if self.mocking:
                #     cat_i = pred_classes[i]
                #     model_i = object_ids[instances.model_indices[i].item()]
                # else:
                #     cat_i, model_i = object_ids[i]
                # try:
                #     sym = next(
                #         a['sym']
                #         for a in self._scene_alignments[scene]
                #         if int(a['catid_cad']) == int(cat_i)
                #         and a['id_cad'] == model_i
                #     )
                # except StopIteration:
                #     sym = "__SYM_NONE"
                    # print("StopIteration encountered")

                # is_dup = (
                #     translation_diff(pred_trans[i], pred_trans[j]) <= NMS_TRANS
                #     and scale_ratio(pred_scale[i], pred_scale[j]) <= NMS_SCALE
                #     and rotation_diff(pred_rot[i], pred_rot[j], sym) <= NMS_ROT
                # )
                is_dup = (
                    translation_diff(pred_trans[i], pred_trans[j]) <= NMS_TRANS
                    and scale_ratio(pred_scale[i], pred_scale[j]) <= NMS_SCALE
                    and rotation_diff(pred_rot[i], pred_rot[j]) <= NMS_ROT
                )
                if is_dup:
                    valid_map[j] = False
            self.results[scene] = instances[valid_map]


    
    def _apply_constraints(self):
        print('INFO: Applying Scan2CAD constraints...', flush=True)
        class_map = self._metadata.thing_dataset_id_to_contiguous_id
        for scene, instances in self.results.items():
            gt_counts = self._scene_counts[scene]
            pred_counts = Counter()
            mask = torch.ones(len(instances), dtype=torch.bool)
            for i, catid in enumerate(instances.pred_classes.tolist()):
                c_name = self._category_manager.get_name(catid)
                # if c_name == 'storagefurniture':
                #     _gt_counts = gt_counts[CAD_TAXONOMY_REVERSE[c_name][0]] + gt_counts[CAD_TAXONOMY_REVERSE[c_name][1]]
                # else:
                cid = CAD_TAXONOMY_REVERSE[c_name]
                _gt_counts = gt_counts[cid]
                if pred_counts[catid] >= _gt_counts:
                    mask[i] = False
                else:
                    pred_counts[catid] += 1
            self.results[scene] = instances[mask]

    def _compute_metrics(self):
        print('INFO: Computing final metrics...', flush=True)

        corrects_per_class = Counter()
        counts_per_class = Counter()
        # for scene, instances in self.results.items():
        for scene, instances in tqdm(self.results.items(), dynamic_ncols=True):
            corrects, counts = self._count_corrects(scene, instances)
            corrects_per_class.update(corrects)
            counts_per_class.update(counts)
        render_nocs(self.noc_tar_trip[:15])

        if self.info_file != '':
            print('Writing evaluation info to {}...'.format(self.info_file))
            with open(self.info_file, 'w') as f:
                json.dump(dict(self.info_data), f)

        if self._exclude is not None:
            for scene in self._exclude:
                for cat, count in self._scene_counts[scene].items():
                    if cat in counts_per_class:
                        counts_per_class[cat] += count

        # if not self.mocking:
        # corrects_per_class = Counter({
        #     self._category_manager.get_name(k): v
        #     for k, v in corrects_per_class.items()
        # })
        # counts_per_class = Counter({
        #     self._category_manager.get_name(k): v
        #     for k, v in counts_per_class.items()
        # })
        # else:
        corrects_per_class = Counter({
            CAD_TAXONOMY[k]: v
            for k, v in corrects_per_class.items()
        })
        counts_per_class = Counter({
            CAD_TAXONOMY[k]: v
            for k, v in counts_per_class.items()
        })

        accuracies = OrderedDict()
        # import pdb; pdb.set_trace()
        for cat in counts_per_class.keys():
            accuracies[cat] = np.round(
                100 * corrects_per_class[cat] / counts_per_class[cat],
                decimals=1
            )

        print()
        print(tabulate(
            sorted(accuracies.items(), key=lambda x: x[0]),
            tablefmt='github',
            headers=['class', 'accuracy']
        ))

        category_average = np.mean(list(accuracies.values()))
        benchmark_average = np.mean([
            acc for cat, acc in accuracies.items()
            if self._category_manager.is_benchmark_class(cat)
        ])
        instance_average = 100 * (
            sum(corrects_per_class.values()) / sum(counts_per_class.values())
        )

        instance_benchmark_average = 100 * (
            sum(
                val for cat, val in corrects_per_class.items()
                if self._category_manager.is_benchmark_class(cat)
            ) / sum(
                val for cat, val in counts_per_class.items()
                if self._category_manager.is_benchmark_class(cat)
            )
        )

        metrics = OrderedDict({
            'category': np.round(category_average, decimals=1),
            'benchmark': np.round(benchmark_average, decimals=1),
            'instance (all)': np.round(instance_average, decimals=1),
            'instance (benchmark)':
                np.round(instance_benchmark_average, decimals=1)
        })
        print()
        print(tabulate(
            list(metrics.items()),
            tablefmt='github',
            headers=['metric', 'accuracy']
        ))
        print()

        return OrderedDict({self.key_prefix + 'alignment': metrics})

    def _count_corrects(self, scene, instances):
        labels = self._scene_alignments[scene]
        # if not self.mocking:
        #     class_map = self._metadata.thing_dataset_id_to_contiguous_id
        #     labels = [
        #         {**label, 'category_id': class_map[label['catid_cad']]}
        #         for label in labels
        #     ]

        label_counts = Counter()
        # for label in labels:
        #     if not self.mocking:
        #         label_counts[label['category_id']] += 1
        #     else:
        #         label_counts[int(label['catid_cad'])] += 1
        for label in labels:
            label_cid = label['catid_cad'] if label['id_cad'] not in self.storagefurniture_ids else '90000000'
            label_counts[int(label_cid)] += 1

        corrects = Counter()
        covered = [False for _ in labels]
        # max_f1s = [-1. for _ in labels]
        # not_in_trains = [(l['catid_cad'], l['id_cad']) in set(self.train_grids.keys()) for l in labels]
        for i in range(len(instances)):
            pred_trans = instances.pred_translations[i]
            pred_rot = np.quaternion(*instances.pred_rotations[i].tolist())
            pred_scale = instances.pred_scales[i]
            pred_class = instances.pred_classes[i].item()
            c_name = self._category_manager.get_name(pred_class)
            object_ids = self.object_ids[scene]
            if self.object_params:
                object_params = self.object_params[scene]
                pred_params = object_params[i]
            if self.object_comps:
                object_comps = self.object_comps[scene]
                nocs_comp = object_comps[i]
            # if self.mocking:
            #     model_i = object_ids[instances.model_indices[i].item()]
            #     cat_i = pred_class
            #     pred_params = object_params[instances.model_indices[i].item()]
            # else:
            cat_i, model_i = object_ids[i]
            # try:
            #     sym_i= next(
            #         a['sym']
            #         for a in self._scene_alignments[scene]
            #         if int(a['catid_cad']) == int(cat_i)
            #         and a['id_cad'] == model_i
            #     )
            # except StopIteration:
            #     sym_i = "__SYM_NONE"
                # print("StopIteration encountered")
            if not self.exact_ret:
                if self.DEFORM:
                    default_param, vertices_mat, faces, constraint_proj_mat = self.source_info[model_i]
                    curr_param = np.array(pred_params, copy=True)
                    curr_param = np.expand_dims(curr_param, -1)
                    curr_mat, curr_default_param, curr_conn_mat = get_source_info_mesh(vertices_mat, default_param, constraint_proj_mat, curr_param.shape[0])
                    output_vertices = get_shape_numpy(curr_mat, curr_param, curr_default_param.T, connectivity_mat=curr_conn_mat)
                    pred_mesh = Meshes([torch.from_numpy(output_vertices).type(torch.float)], [torch.from_numpy(faces).type(torch.float)])#.to('cuda')
                    # pred_mesh = trimesh.Trimesh(
                    #     vertices=output_vertices,
                    #     faces=faces
                    # )
                    # pred_verts_idx = fps(torch.from_numpy(output_vertices).to('cuda'), ratio=1024/output_vertices.shape[0], random_start=False)
                    # pred_verts = output_vertices[pred_verts_idx.cpu().numpy()]
                    # pred_verts = pred_mesh.sample(1024)

                else:
                    pred_mesh = self.cad_manager_train.model_by_id(cat_i, model_i)
                    # pred_mesh = trimesh.Trimesh(
                    #     vertices=pred_mesh.verts_list()[0].cpu().numpy(),
                    #     faces=pred_mesh.faces_list()[0].cpu().numpy()
                    # )
                    # pred_verts_idx = fps(torch.from_numpy(pred_mesh.vertices).to('cuda'), ratio=1024/pred_mesh.vertices.shape[0], random_start=False)
                    # pred_verts = pred_mesh.vertices[pred_verts_idx.cpu().numpy()]
                    # pred_verts = pred_mesh.sample(1024)
                    # pred_verts = self.cad_manager_train._points[0][cat_i, model_i].cpu().numpy()
                pred_mesh_transformed = transform_mesh(pred_mesh, pred_trans.cpu(), pred_rot, pred_scale.cpu())
            match = None
            for j, label in enumerate(labels):
                if covered[j]:
                    continue
                gt_class = (
                    label['catid_cad']
                ) if label['id_cad'] not in self.storagefurniture_ids else '90000000'
                # if c_name == 'storagefurniture':
                #     print(c_name, gt_class)
                #     print(CAD_TAXONOMY_REVERSE[c_name])
                # if c_name == 'storagefurniture' and gt_class not in CAD_TAXONOMY_REVERSE[c_name]:
                #     continue
                if self._category_manager.get_name(pred_class) != CAD_TAXONOMY[int(gt_class)]:
                    continue

                gt_trans = torch.tensor(label['t'])
                gt_rot = np.quaternion(*label['q'])
                gt_scale = torch.tensor(label['s'])
                # sym_i= next(
                #     a['sym']
                #     for a in self._scene_alignments[scene]
                #     if int(a['catid_cad']) == int(label['catid_cad'])
                #     and a['id_cad'] == label['id_cad']
                # )
                # if sym_i == label['sym']:
                #     angle_diff = rotation_diff(pred_rot, gt_rot, sym_i)
                # else:
                angle_diff = rotation_diff(pred_rot, gt_rot)
                is_correct = (
                    translation_diff(pred_trans, gt_trans) <= TRANS_THRESH
                    and angle_diff <= ROT_THRESH
                    and scale_ratio(pred_scale, gt_scale) <= SCALE_THRESH
                )
                # if is_correct and not self.exact_ret:
                #     mesh.export('tst_deformed.obj')
                #     trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{model_i}/models/model_normalized.obj", force='mesh').export('tst_src.obj')
                #     trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{label['id_cad']}/models/model_normalized.obj", force='mesh').export('tst_tar.obj')
                #     trimesh.PointCloud(nocs_comp[1].T).export('tst_nocs.ply')
                #     exit()
                #     self.noc_tar_trip.append((pred_mesh, nocs_comp[1], label['id_cad']))

                if self.exact_ret:
                    cad_pred = (int(cat_i), model_i)
                    cad_gt = (int(label['catid_cad']), label['id_cad'])
                    is_correct = is_correct and cad_pred == cad_gt
                    # is_correct = is_correct and (model_i in self.ret_lookup[label['id_cad']])
                # elif self.with_grids:
                else:
                    try:
                        # iou = self._voxel_iou(label, pred_ind)
                        # if iou >= 0.4:
                        # print(iou)
                        # print('Passed')
                        gt_mesh = self.cad_manager.model_by_id(label['catid_cad'], label['id_cad'])
                        # metrics = compare_meshes(pred_mesh, gt_mesh, thresholds=[0.3])
                        metrics = compare_meshes(
                            pred_mesh_transformed.to('cuda'), 
                            transform_mesh(gt_mesh, gt_trans, gt_rot, gt_scale).to('cuda'), 
                            thresholds=[0.5],
                            # scale='gt-1'
                        )
                        f1_score = metrics['F1@0.500000']

                        # gt_mesh = trimesh.Trimesh(
                        #     vertices=gt_mesh.verts_list()[0],
                        #     faces=gt_mesh.faces_list()[0]
                        # )
                        # gt_verts = self.cad_manager._points[0][label['catid_cad'], label['id_cad']].cpu().numpy()
                        # gt_verts = fps(gt_mesh.vertices, ratio=2048/gt_mesh.vertices.shape[0])
                        # gt_verts = gt_mesh.sample(10000)
                        # pred_verts = self.cad_manager_train._points[0][cat_i, model_i].cpu().numpy()
                        # pred_verts,_,_ = trimesh.proximity.closest_point(pred_mesh, pred_verts)
                        # emd_score = emd(pred_verts, gt_verts)
                    except KeyError:
                        iou = 0.0
                        print('failed')
                    # is_correct = is_correct and iou >= VOXEL_IOU_THRESH
                    # is_correct = is_correct and f1_score >= 50.
                    is_correct = f1_score >= 50.
                    # is_correct = is_correct and emd_score <= 0.065
                
                # if is_correct and not self.exact_ret:
                #     pred_verts = sample_points_from_meshes(pred_mesh, 1024).cpu().numpy()[0]
                #     gt_verts = sample_points_from_meshes(gt_mesh, 1024).cpu().numpy()[0]
                #     emd_scores[int(label['catid_cad'])] += emd(pred_verts, gt_verts)

                if is_correct:
                    corrects[int(gt_class)] += 1
                    covered[j] = True
                    # ret_covered[k] = True
                    match = {'index': j, 'label': label}
                    break

            if self.info_file != '':
                self.info_data[scene].append({
                    'id_cad': model_i,
                    'catid_cad': cat_i,
                    'match': match,
                    't': pred_trans.tolist(),
                    'q': quaternion.as_float_array(pred_rot).tolist(),
                    's': pred_scale.tolist()
                })

        # print([a[1] for a in zip(not_in_trains, max_f1s) if not a[0]])
        return corrects, label_counts

    def _voxel_iou(self, label, pred_ind):
        # import pdb; pdb.set_trace()
        
        gt_ind = self.val_grid_data[label['catid_cad'], label['id_cad']]
        # gt_ind = self.train_grids[label['catid_cad'], label['id_cad']]
        
        pred_ind_1d = np.unique(np.ravel_multi_index(
            multi_index=(pred_ind[:, 0], pred_ind[:, 1], pred_ind[:, 2]),
            dims=(32, 32, 32)
        ))

        gt_ind_1d = np.unique(np.ravel_multi_index(
            multi_index=(gt_ind[:, 0], gt_ind[:, 1], gt_ind[:, 2]),
            dims=(32, 32, 32)
        ))
        inter = np.intersect1d(pred_ind_1d, gt_ind_1d).size
        union = pred_ind_1d.size + gt_ind_1d.size - inter
        return inter / union


def eval_csv(
    dataset_name: str,
    csv_path: str,
    full_annot=None,
    grid_file=None,
    exact_ret=False,
    prefix: str = '',
    info_file=''
) -> OrderedDictType[str, Dict[str, float]]:

    # FIXME: relative path!
    eval_path = __file__
    for i in range(4):  # -> eval -> roca -> network
        eval_path = os.path.dirname(eval_path)
    eval_path = os.path.join(eval_path, 'metadata', 'scannetv2_val.txt')
    with open(eval_path) as f:
        val_scenes = set(ln.strip() for ln in f)

    data = read_csv(csv_path)
    scenes = data['id_scan'].unique()
    exclude = val_scenes.difference(scenes)
    evaluator = Vid2CADEvaluator(
        dataset_name,
        full_annot=full_annot,
        mocking=True,
        exclude=exclude,
        grid_file=grid_file,
        exact_ret=exact_ret,
        key_prefix=prefix,
        info_file=info_file
    )
    evaluator.reset()

    print('INFO: Processing outputs...')
    for i, scene in enumerate(scenes):
        # print('{} / {}'.format(i, len(scenes)))
        scene_data = data[data['id_scan'] == scene]

        pred_trans = np.hstack([
            scene_data['tx'][:, None],
            scene_data['ty'][:, None],
            scene_data['tz'][:, None]
        ])
        pred_rot = np.hstack([
            scene_data['qw'][:, None],
            scene_data['qx'][:, None],
            scene_data['qy'][:, None],
            scene_data['qz'][:, None]
        ])
        pred_scale = np.hstack([
            scene_data['sx'][:, None],
            scene_data['sy'][:, None],
            scene_data['sz'][:, None]
        ])
        pred_catids = np.asarray(scene_data['objectCategory']).tolist()
        scores = np.asarray(scene_data['object_score'])

        model_list = scene_data['alignedModelId'].tolist()
        model_indices = torch.arange(len(model_list), dtype=torch.long)

        instances = Instances(IMAGE_SIZE)
        instances.pred_translations = torch.from_numpy(pred_trans).float()
        instances.pred_rotations = torch.from_numpy(pred_rot).float()
        instances.pred_scales = torch.from_numpy(pred_scale).float()
        instances.pred_classes = torch.tensor(pred_catids)
        instances.scores = torch.from_numpy(scores).float()
        instances.model_indices = model_indices
        instances = instances[instances.scores.argsort(descending=True)]

        evaluator.process_mock(scene, instances, model_list)

    return evaluator.evaluate_mock()

def create_grid_points_from_xyz_bounds(min_x, max_x, min_y, max_y ,min_z, max_z, res):
    x = np.linspace(min_x, max_x, res)
    y = np.linspace(min_y, max_y, res)
    z = np.linspace(min_z, max_z, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=False)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list

def voxelize_mesh(mesh, grid_size=32):
    indices = product(range(grid_size), range(grid_size), range(grid_size))
    indices = np.array(list(indices), dtype=np.int32)
    grid = local_voxelize(
        mesh=mesh,
        point=np.zeros(3),
        pitch=(1 / grid_size),
        radius=(grid_size // 2),
        fill=False
    )
    grid = np.asarray(grid.matrix).transpose((2, 1, 0))
    grid = grid[:grid_size, :grid_size, :grid_size]

    return indices[grid.reshape(-1)]

def render_nocs(noc_trips):
    def image_grid(imgs, rows, cols):
        w, h = imgs[0].size
        dw, dh = (256,256)
        grid = Image.new('RGB', size=(cols*dw, rows*dh))
        
        for i, img in enumerate(imgs):
            im = img.crop(((w-h)//2, 0, w-((w-h)//2), h))
            im = im.resize((dw,dh), Image.ANTIALIAS)
            grid.paste(im, box=(i%cols*dw, i//cols*dh))
        return grid

    def render_image_from_mesh(_mesh, renderer, device):
        if type(_mesh) is str:
            vertices, faces, _ = load_obj(_mesh, device=device)
            mesh = Meshes([vertices], [faces.verts_idx])
        elif type(_mesh) is Meshes:
            mesh = _mesh
            vertices = mesh.verts_list()[0]
        else:
            vertices, faces = (torch.from_numpy(_mesh.vertices).to(device).type(torch.float), torch.from_numpy(_mesh.faces).to(device).type(torch.float))
            mesh = Meshes([vertices], [faces])

        color = torch.ones(1, vertices.size(0), 3, device=device)
        mesh.textures = TexturesVertex(verts_features=color)
        img = renderer(mesh.to(device))
        img = (img.detach().cpu().numpy()[0] * 255).astype('uint8')

        return img

    def render_image_from_pc(pointcloud, renderer, device):
        verts = torch.Tensor(pointcloud).to(device)
        if verts.shape[0] == 3:
            verts = verts.transpose(0,1)
        try:
            point_cloud = Pointclouds(points=[verts], features=[torch.ones_like(verts)])
        except ValueError as e:
            print(e)
            print(verts.shape)
            return np.zeros(256,256,3)
        img = renderer(point_cloud)
        img = (img.detach().cpu().numpy()[0] * 255).astype('uint8')

        return img
    
    if not noc_trips:
        print("List is empty, won't render!")
        return

    device = 'cuda'
    renderer_params = {
        'image_size': 256,
        'camera_dist': 2.5,
        'elevation': 0,
        'azim_angle': 0,
    }
    R, T = look_at_view_transform(3)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras)
    )
    pc_renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=PointsRasterizationSettings(
        image_size=256, 
        radius = 0.003,
        points_per_pixel = 10)),
        compositor=AlphaCompositor()
    )

    out_path = "img_grid.png"

    to_grid_imgs = []
    for comp, nocs, target in noc_trips:
        # comp_img = render_image_from_pc(comp, pc_renderer, device)
        comp_img = render_image_from_mesh(comp, mesh_renderer, device)
        noc_img = render_image_from_pc(nocs, pc_renderer, device)
        tar_img = render_image_from_mesh(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{target}/models/model_normalized.obj", mesh_renderer, device)
        to_grid_imgs.extend([Image.fromarray(noc_img).convert('RGB'), Image.fromarray(comp_img).convert('RGB'), Image.fromarray(tar_img).convert('RGB')])

    grid = image_grid(to_grid_imgs, rows=len(noc_trips), cols=3)
    grid.save(out_path)
    grid.close()
    for img in to_grid_imgs: img.close()

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def emd(p,q):
    d = cdist(p, q)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(p), len(q))