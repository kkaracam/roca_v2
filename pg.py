import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import pickle
import numpy as np
# np.random.seed(0)

import json
import cv2
import trimesh
import h5py
import random
from joblib import Parallel, delayed
import multiprocessing
from glob import glob
from tqdm import tqdm
from itertools import product
from trimesh.voxel.creation import local_voxelize
import torch
from torch import nn
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.renderer import(
    look_at_view_transform,
    PerspectiveCameras,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.ops import sample_farthest_points
from PIL import Image

sys.path.append('network')
from network.roca.utils.linalg import decompose_mat4, make_M_from_tqs
from network.roca.modeling.retrieval_head.joint_retrieve_deform_ops import (
    get_model,
    get_source_info,
    get_shape,
    get_source_latent_codes_fixed,
    compute_mahalanobis
)
from network.roca.modeling.retrieval_head.joint_retrieve_deform_modules import (
    TargetEncoder,
    TargetDecoder,
    ParamDecoder2
)

from network.roca.utils.ap import compare_meshes
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from torch_cluster import fps

import logging
logging.getLogger('trimesh').setLevel('ERROR')

# sys.path.append("/home/karacam/Thesis/joint_learning_retrieval_deformation")
# from img_grid import mesh_renderer, render_image_from_mesh

def joint_input_data(split = 'train'):
    cad_file = f'/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_{split}_cads.pkl'
    grids_file = f'/mnt/noraid/karacam/Roca/Data/Dataset/{split}_grids_32.pkl'
    points_file = f'/mnt/noraid/karacam/Roca/Data/Dataset/points_{split}.pkl'
    instances_file = f'/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_instances_{split}.json'
    shape2part_file = '/mnt/noraid/karacam/ThesisData/data/shape2part.json'
    shapenet_chair_synsetid = '03001627'
    out_dir = '/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/'

    joint_cads_file = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data/roca_sources_part_thresh_32_1024p/chair/h5'

    with open(cad_file, 'rb') as f:
        models = pickle.load(f)
    with open(grids_file, 'rb') as f:
        grids = pickle.load(f)
    with open(points_file, 'rb') as f:
        points = pickle.load(f)
    with open(instances_file, 'rb') as f:
        instances = json.load(f)
    partnet_ids = [e.split('_')[0] for e in os.listdir(joint_cads_file)]
    with open(shape2part_file, 'rb') as f:
        shape2part = json.load(f)
    # joint_cads = [e for e in shape2part if shape2part[e] in partnet_ids]
    # joint_cads = [e['id_cad'] for e in models if e['id_cad'] in joint_cads]

    # print("Creating models...")
    # roca_joint_chair_models = [e for e in models if e['id_cad'] in joint_cads]
    # for e in roca_joint_chair_models:
    #     e['category_id'] = 0
    # print("Creating points...")
    # roca_joint_chair_points = [e for e in points if e['id_cad'] in joint_cads]
    # for e in roca_joint_chair_points:
    #     e['category_id'] = 0
    print("Creating models...")
    roca_joint_chair_models = [e for e in models if e['catid_cad'] == shapenet_chair_synsetid]
    for e in roca_joint_chair_models:
        e['category_id'] = 0
    print("Creating points...")
    roca_joint_chair_points = [e for e in points if e['catid_cad'] == shapenet_chair_synsetid]
    for e in roca_joint_chair_points:
        e['category_id'] = 0

    roca_joint_chair_grids = {}
    # for cad_id in joint_cads:
    #     roca_joint_chair_grids[(shapenet_chair_synsetid, cad_id)] = grids[(shapenet_chair_synsetid, cad_id)]
    for k in grids:
        if k[0] == shapenet_chair_synsetid:
            roca_joint_chair_grids[k] = grids[k]

    # for e in instances['categories']:
    #     if e['name'] == 'chair':
    #         cat_id = e['id']
    #         break

    roca_joint_chair_instances = {}
    # roca_joint_chair_instances['categories'] = instances['categories']
    roca_joint_chair_instances['categories'] = [{"name": "chair", "id": 0}]
    annots = []
    with open('/home/karacam/Thesis/ROCA/metadata/scenes.txt', 'r') as scns_f:
        joint_chair_scenes = {e: False for e in scns_f.read().splitlines()}
    print("Creating annotations...")
    # for e in instances['annotations']:
    #     if e['model']["id_cad"] in joint_cads:
    #         e['category_id'] = 0
    #         annots.append(e)
    #         joint_chair_scenes[e["scene_id"]] = True
    for e in instances['annotations']:
        if e['model']["catid_cad"] == shapenet_chair_synsetid:
            e['category_id'] = 0
            annots.append(e)
            joint_chair_scenes[e["scene_id"]] = True
    roca_joint_chair_instances['annotations'] = annots
    print("Creating images...")
    roca_joint_chair_instances['images'] = [e for e in instances['images'] if joint_chair_scenes[e["scene_id"]]]
    # joint_chair_scenes = []
    # print("Creating annotations...")
    # for e in instances['annotations']:
    #     if e['model']["id_cad"] in joint_cads:
    #         e['category_id'] = 0
    #         annots.append(e)
    #         joint_chair_scenes.append(e["scene_id"])
    # roca_joint_chair_instances['annotations'] = annots
    # print("Creating images...")
    # roca_joint_chair_instances['images'] = [e for e in instances['images'] if e["scene_id"] in joint_chair_scenes]

    print("Dumping...")
    with open(os.path.join(out_dir, f'scan2cad_{split}_cads.pkl'), 'wb') as f:
        pickle.dump(roca_joint_chair_models, f)
    with open(os.path.join(out_dir, f'{split}_grids_32.pkl'), 'wb') as f:
        pickle.dump(roca_joint_chair_grids, f)
    with open(os.path.join(out_dir, f'points_{split}.pkl'), 'wb') as f:
        pickle.dump(roca_joint_chair_points, f)
    with open(os.path.join(out_dir, f'scan2cad_instances_{split}.json'), 'w') as f:
        json.dump(roca_joint_chair_instances, f)


    print('Points size: ', len(points))
    print('Models size: ', len(models))
    print('Joint Points size: ', len(roca_joint_chair_points))
    print('Joint Models size: ', len(roca_joint_chair_models))
    print('Images size: ', len(instances['images']))
    print('Joint Images size: ', len(roca_joint_chair_instances['images']))
    print('Annotations size: ', len(instances['annotations']))
    print('Joint Annotations size: ', len(roca_joint_chair_instances['annotations']))
    # print(len(roca_joint_chair_grids.keys()))
    if split  == "val":
        print("Val split:")
        scenes_file = f'/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_val_scenes.json'
        with open(scenes_file, 'rb') as f:
            scenes = json.load(f)
        chair_scenes = {}
        for scene_id in scenes:
            chair_vals = [e for e in scenes[scene_id] if e['category_id'] == 5]
            if chair_vals:
                for e in chair_vals:
                    e['category_id'] = 0
                chair_scenes[scene_id] = chair_vals
        with open(os.path.join(out_dir, f'scan2cad_val_scenes.json'), 'w') as f:
            json.dump(chair_scenes, f)
        print(len(scenes))
        print(len(chair_scenes))

def process_400k_custom(split = 'train'):
    with open("/mnt/noraid/karacam/AlignmentNew/scan2cad_matches.normalized.json", 'r') as jf:
        # keys: scenes
        #   dict_keys(['object_id', 'cad_id', 'alignment'])
        annos_400k =json.load(jf)

    with open("/mnt/noraid/karacam/AlignmentNew/cad_scales.json", 'r') as jf:
        cad_scales =json.load(jf)

    with open(f"/mnt/noraid/karacam/AlignmentNew/SlamDataset_Normalized/scannet_instances_{split}.json", 'r') as jf:
        # dict_keys(['images', 'annotations', 'categories'])
        #   dict_keys(['id', 'bbox', 'area', 'segmentation', 'iscrowd', 'image_id', 'category_id', 'object_id', 'alignment'])
        instances_split_400k =json.load(jf)

    # with open(f"/home/karacam/Thesis/ROCA/Data/Dataset/scan2cad_instances_{split}.json", 'r') as jf:
    #     # dict_keys(['images', 'annotations', 'categories'])
    #     #   dict_keys(['id', 'bbox', 'area', 'segmentation', 'iscrowd', 'image_id', 'category_id', 'scene_id', 'is_cad', 'intrinsics', 'alignment_id', 'q', 's', 't', 'sym', 'model'])
    #     instances_split_25k =json.load(jf)
    with open("/home/karacam/Thesis/ROCA/Data/full_annotations.json") as f:
        # dict_keys(['trs', 'n_aligned_models', 'aligned_models', 'id_scan'])
        # 'aligned_models': dict_keys(['trs', 'bbox', 'center', 'sym', 'id_cad', 'catid_cad', 'keypoints_cad', 'keypoints_scan'])
        annos_25k = json.load(f)
    sym_dicts = {}
    for e in annos_25k:
        sym_dicts[e['id_scan']] = {}
        for m in e['aligned_models']:
            sym_dicts[e['id_scan']][f"{m['catid_cad']}_{m['id_cad']}"] = m['sym']
    instances_custom = {}
    instances_custom['categories'] = instances_split_400k['categories']
    instances_custom['images'] = instances_split_400k['images']
    instances_custom['annotations'] = []
    imgs = {}
    for e in instances_custom['images']:
        ofname = e['file_name'].split('/')
        e['scene_id'] = ofname[2]
        e['file_name'] = "/".join(ofname[2:])
        imgs[e['id']] = [e['scene_id'], e['file_name']]

    itr = 1
    lng = len(instances_split_400k['annotations'])

    for e in instances_split_400k['annotations']:
        print("Processing ", itr, " of ", lng, "...")
        d = {}
        d['id'] = e['id']
        d['bbox'] = e['bbox']
        d['area'] = e['area']
        d['segmentation'] = e['segmentation']
        d['iscrowd'] = e['iscrowd']
        d['image_id'] = e['image_id']
        d['category_id'] = e['category_id']
        d['scene_id'] = imgs[d['image_id']][0]
        with open(f"/mnt/noraid/karacam/AlignmentNew/Resized400k/tasks/scannet_frames_25k/{d['scene_id']}/intrinsics_color.txt", 'r') as inf:
            ins = [list(map(float, e.split(' '))) for e in inf.readlines()]
        d['intrinsics'] = ins
        d['alignment_id'] = e['object_id']
        with open(f"/mnt/noraid/karacam/AlignmentNew/Resized400k/tasks/scannet_frames_25k/{imgs[d['image_id']][0]}/pose/{imgs[d['image_id']][1].split('/')[-1].split('.')[0]}.txt", 'r') as f:
            T = f.readlines()
            T = np.asarray([np.array(t.split(), np.float32) for t in T])
        # d['t'] = e['alignment'][0]
        # d['q'] = e['alignment'][1]
        for s in annos_400k[d['scene_id']]:
            if s['object_id'] == e['object_id']:
                d['sym'] = sym_dicts[d['scene_id']][f"{s['cad_id'][0]}_{s['cad_id'][1]}"]
                scls = cad_scales[f"{s['cad_id'][0]}_{s['cad_id'][1]}"][0]
                # d['s'] = [e['alignment'][2][0] * scls[0], e['alignment'][2][1] * scls[1], e['alignment'][2][2] * scls[2]]
                d['t'], d['q'], d['s'] = map(list, decompose_mat4(np.linalg.inv(T) @ make_M_from_tqs(e['alignment'][0], e['alignment'][1], [e['alignment'][2][0] * scls[0], e['alignment'][2][1] * scls[1], e['alignment'][2][2] * scls[2]])))
                d['model'] = {}
                d['model']['scene_id'] = d['scene_id']
                d['model']['catid_cad'] = s['cad_id'][0]
                d['model']['id_cad'] = s['cad_id'][1]
                break
        instances_custom['annotations'].append(d)
        itr += 1

    print("Dumping...")
    with open(f"/home/karacam/Thesis/ROCA/Data/Dataset/400k_instances_{split}.json", 'w') as jf:
        # dict_keys(['images', 'annotations', 'categories'])
        #   dict_keys(['id', 'bbox', 'area', 'iscrowd', 'image_id', 'category_id', 'scene_id', 'is_cad', 'intrinsics', 'alignment_id', 'q', 's', 't', 'sym', 'model'])
        json.dump(instances_custom, jf)

def proccess_annotations():
    with open("/mnt/noraid/karacam/AlignmentNew/scan2cad_matches.normalized.json", 'r') as jf:
        # keys: scenes
        #   dict_keys(['object_id', 'cad_id', 'alignment'])
        #     # dict_keys(['trs', 'n_aligned_models', 'aligned_models', 'id_scan'])
        #     # 'aligned_models': dict_keys(['trs', 'bbox', 'center', 'sym', 'id_cad', 'catid_cad', 'keypoints_cad', 'keypoints_scan'])
        annos_400k =json.load(jf)
    with open("/mnt/noraid/karacam/AlignmentNew/cad_scales.json", 'r') as jf:
        cad_scales =json.load(jf)
    with open("/home/karacam/Thesis/ROCA/Data/full_annotations.json") as f:
        # dict_keys(['trs', 'n_aligned_models', 'aligned_models', 'id_scan'])
        # 'aligned_models': dict_keys(['trs', 'bbox', 'center', 'sym', 'id_cad', 'catid_cad', 'keypoints_cad', 'keypoints_scan'])
        annos_25k = json.load(f)
    # sym_dicts = {}
    # for e in annos_25k:
    #     sym_dicts[e['id_scan']] = {}
    #     for m in e['aligned_models']:
    #         sym_dicts[e['id_scan']][f"{m['catid_cad']}_{m['id_cad']}"] = m['sym']
    new_annots = []
    for e in annos_25k:
        scene = e['id_scan']
        annot = {}
        annot['trs'] = e['trs']
        annot['n_aligned_models'] = len(annos_400k[scene])
        aligned_models = []
        for m in annos_400k[scene]:
            for _m in e['aligned_models']:
                if _m['id_cad'] == m['cad_id'][1]:
                    aligned_models.append(_m)
                    break
        annot['aligned_models'] = aligned_models
        annot['id_scan'] = scene
        new_annots.append(annot)
    
    # for scene in annos_400k:
    #     annot = {}
    #     annot['trs'] = {'translation': [0.,0.,0.], 'rotation':[1.,0.,0.,0.], 'scale':[1.,1.,1.]}
    #     annot['n_aligned_models'] = len(annos_400k[scene])
    #     aligned_models = []
    #     for m in annos_400k[scene]:
    #         am = {}
    #         am['trs'] = {'translation': m['alignment'][0], 'rotation': m['alignment'][1], 'scale': list(np.multiply(m['alignment'][2], cad_scales[f"{m['cad_id'][0]}_{m['cad_id'][1]}"][0]))}
    #         am['id_cad'] = m['cad_id'][1]
    #         am['catid_cad'] = m['cad_id'][0]
    #         am['sym'] = sym_dicts[scene][f"{m['cad_id'][0]}_{m['cad_id'][1]}"]
    #         aligned_models.append(am)
    #     annot['aligned_models'] = aligned_models
    #     annot['id_scan'] = scene
    #     new_annots.append(annot)
    with open("/home/karacam/Thesis/ROCA/Data/full_annotations_400k.json", 'w') as f:
        json.dump(new_annots, f)

def make_mock_instances(split='val'):
    # instances_file = f'/home/karacam/Thesis/ROCA/scan2cad_instances_train.json'
    instances_file = f'/mnt/noraid/karacam/Roca/Data/Dataset/Custom5Class/scan2cad_instances_{split}.json'
    with open(instances_file, 'rb') as f:
        instances = json.load(f)

    mock_ins = {"categories": instances['categories'], "images":np.random.choice(instances['images'], size=200, replace=False).tolist()}
    img_ids = [i['id'] for i in mock_ins['images']]
    mock_ins['annotations'] = [a for a in instances['annotations'] if a['image_id'] in img_ids]

    with open(f'/mnt/noraid/karacam/Roca/Data/Dataset/Custom5Class/scan2cad_instances_{split}_mock.json', 'w') as f:
        json.dump(mock_ins, f)

def compare_ious(oracle_retrieval=False):
    def get_source_latent_codes_encoder(source_labels, SOURCE_MODEL_INFO, retrieval_encoder, device):
        # print("Using encoder to get source latent codes.")
        source_points = []

        # start_tm = time.time()
        for source_label in source_labels:
            src_points = SOURCE_MODEL_INFO[source_label]["points"]	
            source_points.append(src_points)
        # ret_tm = time.time()
        # print("Load from labels time: ", ret_tm - start_tm)
        # print("Num labels: ", source_labels.shape)
        source_points = np.array(source_points)
        source_points = torch.from_numpy(source_points).to(device, dtype=torch.float)

        src_latent_codes = retrieval_encoder(source_points)
        # print("Retrieval encoding time: ", time.time() - ret_tm)
        return src_latent_codes

    JOINT_BASE_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/'
    JOINT_SRC_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data/roca_sources_part_thresh_32_1024p_v2/chair/h5'
    JOINT_MODEL_PATH = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/log/chair519_1024p_v2_reg/model.pth'
    filename_pickle = JOINT_BASE_DIR+"data/generated_datasplits/chair_519_roca_v2.pickle"
    with open(filename_pickle, 'rb') as handle:
        sources = pickle.load(handle)['sources']
    src_data_fol = JOINT_SRC_DIR
    MAX_NUM_PARAMS = -1
    MAX_NUM_PARTS = -1
    ALPHA = 0.1
    K = 5
    device = 'cuda'
    SOURCE_MODEL_INFO = []
    print("Loading joint sources...")
    with open("/home/karacam/Thesis/joint_learning_retrieval_deformation/shape2part_ext.json", 'r') as f:
        shape2part = json.load(f)
    part2shape = dict([(value, key) for key, value in shape2part['chair']['model_names'].items()])

    for i in range(len(sources)):
        source_model = sources[i]
        src_filename = str(source_model) + "_leaves.h5"

        default_param, points, point_labels, points_mat, \
                    constraint_mat,	constraint_proj_mat	= get_model(src_data_fol + '/' + src_filename, constraint=True)

        curr_source_dict = {}
        curr_source_dict["default_param"] = default_param
        curr_source_dict["points"] = points
        curr_source_dict["point_labels"] = point_labels
        curr_source_dict["points_mat"] = points_mat
        curr_source_dict["model_id"] = (shape2part['chair']['synsetid'], part2shape[str(source_model)])

        curr_source_dict["constraint_mat"] = constraint_mat
        curr_source_dict["constraint_proj_mat"] = constraint_proj_mat
        _, curr_source_dict["vertices_mat"], curr_source_dict["faces"], _ = get_model(os.path.join(src_data_fol, src_filename), pred=True)

        # Get number of parts of the model
        num_parts = len(np.unique(point_labels))
        curr_source_dict["num_parts"] = num_parts

        curr_num_params = default_param.shape[0]
        if (MAX_NUM_PARAMS < curr_num_params):
            MAX_NUM_PARAMS = curr_num_params
            MAX_NUM_PARTS = int(MAX_NUM_PARAMS/6)

        SOURCE_MODEL_INFO.append(curr_source_dict)
    print("Done loading joint sources.")
    embedding_size = 6
    TARGET_LATENT_DIM = 256
    SOURCE_LATENT_DIM = 256
    PART_LATENT_DIM = 32
    # with torch.no_grad():
    target_encoder = TargetEncoder(
        TARGET_LATENT_DIM,
        3,
    )
    decoder_input_dim = TARGET_LATENT_DIM + SOURCE_LATENT_DIM + PART_LATENT_DIM
    param_decoder = ParamDecoder2(decoder_input_dim, 256, embedding_size)
    # param_decoder.to(device, dtype=torch.float)
    retrieval_encoder = TargetEncoder(
        TARGET_LATENT_DIM,
        3,
    )
    # embed_loss = nn.TripletMarginLoss(margin=MARGIN)

    _model = torch.load(JOINT_MODEL_PATH)
    target_encoder.load_state_dict(_model["target_encoder"])
    target_encoder.to(device)
    target_encoder.eval()

    param_decoder.load_state_dict(_model["param_decoder"])
    param_decoder.to(device)
    param_decoder.eval()

    retrieval_encoder.load_state_dict(_model["retrieval_encoder"])
    retrieval_encoder.to(device)
    retrieval_encoder.eval()

    SOURCE_LATENT_CODES = _model["source_latent_codes"].detach()
    SOURCE_PART_LATENT_CODES = [_x.detach() for _x in _model["part_latent_codes"]]
    SOURCE_VARIANCES = _model["source_variances"].detach()

    source_labels = torch.arange(len(SOURCE_MODEL_INFO))#.repeat(10)
    src_mats, src_default_params, src_connectivity_mat = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity=True)
    src_latent_codes = torch.gather(SOURCE_LATENT_CODES, 0, source_labels.to(device).unsqueeze(-1).repeat(1,SOURCE_LATENT_CODES.shape[-1]))

    mat = torch.stack([mat.to(device, dtype=torch.float) for mat in src_mats])#.to(device, dtype=torch.float)
    def_param = torch.stack([def_param.to(device, dtype=torch.float) for def_param in src_default_params])#.to(device, dtype=torch.float)
    conn_mat = torch.stack([conn_mat.to(device, dtype=torch.float) for conn_mat in src_connectivity_mat])
    with torch.no_grad():
        ret_src_latent_codes = []
        num_sets = 20
        interval = int(len(source_labels)/num_sets)

        for j in range(num_sets):
            if (j==num_sets-1):
                curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:], SOURCE_MODEL_INFO, retrieval_encoder, device)
            else:
                curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:(j+1)*interval], SOURCE_MODEL_INFO, retrieval_encoder, device)
            ret_src_latent_codes.append(curr_src_latent_codes)

        ret_src_latent_codes = torch.cat(ret_src_latent_codes)
    
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/points_val.pkl', 'rb') as f:
        points_val = pickle.load(f)
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/val_grids_32.pkl', 'rb') as f:
        grids_val = pickle.load(f)
    assert len({ps['id_cad'] for ps in points_val}) == len(points_val)
    tar_pcs = {(ps['catid_cad'],ps['id_cad']):ps['points'] for ps in points_val}

    ious = []
    exact = 0
    for cat_id, model_id in tqdm(tar_pcs):
        # tar_pc_np = mesh.sample(1024)
        tar_pc = tar_pcs[cat_id, model_id]
        # tar_pc = tar_pc[tar_pc[:,0] > 0]
        try:
            with open(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{model_id}/models/model_normalized.json", 'r') as jf:
                _meta = json.load(jf)
            # tar_mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{model_id}/models/model_normalized.obj", force='mesh')
        except FileNotFoundError:
            print("MODEL NOT FOUND!?!?")

        # m_max = np.asarray(_meta['max'])
        # m_min = np.asarray(_meta['min'])
        # diag = m_max - m_min
        # center = (m_max + m_min)/2
        # centroid = np.asarray(_meta['centroid'])
        # tar_pc = tar_pc + (centroid - center) / np.linalg.norm(diag)
        tar_pc = torch.from_numpy(tar_pc).unsqueeze(0).to(device, dtype=torch.float)
        tar_pc.requires_grad = False

        # assert tar_pc.shape == (1,1024,3)
        
        if oracle_retrieval:
            target_latent_codes = target_encoder(tar_pc)
            target_latent_codes = target_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
            target_latent_codes = target_latent_codes.view(-1, target_latent_codes.shape[-1])
            x_repeated = tar_pc.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1,1)
            x_repeated = x_repeated.view(-1, x_repeated.shape[-2], x_repeated.shape[-1])
            concat_latent_code = torch.cat((src_latent_codes, target_latent_codes), dim=1)
            all_params = []
            for j in range(concat_latent_code.shape[0]):
                curr_num_parts = SOURCE_MODEL_INFO[source_labels[j]]["num_parts"]
                curr_code = concat_latent_code[j]
                curr_code_repeated = curr_code.view(1,curr_code.shape[0]).repeat(curr_num_parts, 1)
                
                part_latent_codes = SOURCE_PART_LATENT_CODES[source_labels[j]]

                full_latent_code = torch.cat((curr_code_repeated, part_latent_codes), dim=1)

                params = param_decoder(full_latent_code, use_bn=False)

                ## Pad with extra zero rows to cater to max number of parameters
                if (curr_num_parts < MAX_NUM_PARTS):
                    dummy_params = torch.zeros((MAX_NUM_PARTS-curr_num_parts, embedding_size), dtype=torch.float, device=device)
                    params = torch.cat((params, dummy_params), dim=0)

                params = params.view(-1, 1)
                all_params.append(params)

            params = torch.stack(all_params)
            output_pcs = get_shape(mat, params, def_param, ALPHA, connectivity_mat=conn_mat)
            cd_loss, _ = chamfer_distance(output_pcs, x_repeated, batch_reduction=None)
            output_pcs = output_pcs.view(len(SOURCE_MODEL_INFO), tar_pc.shape[0], tar_pc.shape[1], tar_pc.shape[2])
            cd_loss = cd_loss.view(len(SOURCE_MODEL_INFO), -1)#.detach().item()

            cd_loss_sorted, sorted_by_cd = torch.sort(cd_loss, dim=0)
            cd_retrieved = sorted_by_cd[:K,:]
            retrieved_idx = [SOURCE_MODEL_INFO[smi]['model_id'] for smi in cd_retrieved]
            ret_id = shape2part['chair']['model_names'][retrieved_idx[0][1]]
            pred_params = params[cd_retrieved[0]].squeeze(0).detach().cpu().numpy()
            
            # default_param, vertices_mat, faces, constraint_proj_mat = get_model(os.path.join(src_data_fol, ret_id+'_leaves.h5'),pred = True)
            curr_param = np.expand_dims(pred_params, -1)
            curr_mat, curr_default_param, curr_conn_mat = get_source_info_mesh(SOURCE_MODEL_INFO[cd_retrieved[0]]["vertices_mat"], SOURCE_MODEL_INFO[cd_retrieved[0]]["default_param"], SOURCE_MODEL_INFO[cd_retrieved[0]]["constraint_proj_mat"], curr_param.shape[0])
            output_vertices = get_shape_numpy(curr_mat, curr_param, curr_default_param.T, connectivity_mat=curr_conn_mat)
            mesh = trimesh.Trimesh(
                vertices=output_vertices,
                faces=SOURCE_MODEL_INFO[cd_retrieved[0]]["faces"]
            )
        
        else:
            retrieval_latent_codes = retrieval_encoder(tar_pc)
            retrieval_latent_codes = retrieval_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
            # retrieval_latent_codes = retrieval_latent_codes.view(-1, retrieval_latent_codes.shape[-1])	
            retrieval_latent_codes = retrieval_latent_codes.view(len(SOURCE_MODEL_INFO), -1, TARGET_LATENT_DIM)
            with torch.no_grad():
                source_labels = source_labels
                _src_latent_codes = ret_src_latent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

                src_variances = get_source_latent_codes_fixed(source_labels, SOURCE_VARIANCES, device=device)
                src_variances = src_variances.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

            distances = compute_mahalanobis(retrieval_latent_codes, _src_latent_codes, src_variances, activation_fn=torch.sigmoid)
            sorted_indices = torch.argsort(distances, dim=0)
            retrieved_idx = sorted_indices[0,:]

            target_latent_codes = target_encoder(tar_pc)
            concat_latent_code = torch.cat((src_latent_codes[retrieved_idx], target_latent_codes), dim=1)

            curr_num_parts = SOURCE_MODEL_INFO[retrieved_idx]["num_parts"]
            curr_code = concat_latent_code[0]
            curr_code_repeated = curr_code.view(1,curr_code.shape[0]).repeat(curr_num_parts, 1)
            
            part_latent_codes = SOURCE_PART_LATENT_CODES[retrieved_idx]

            full_latent_code = torch.cat((curr_code_repeated, part_latent_codes), dim=1)

            params = param_decoder(full_latent_code, use_bn=False)

            ## Pad with extra zero rows to cater to max number of parameters
            if (curr_num_parts < MAX_NUM_PARTS):
                dummy_params = torch.zeros((MAX_NUM_PARTS-curr_num_parts, embedding_size), dtype=torch.float, device=device)
                params = torch.cat((params, dummy_params), dim=0)

            params = params.view(-1, 1)
            pred_params = params.detach().cpu().numpy()
            curr_param = np.expand_dims(pred_params, -1)
            curr_mat, curr_default_param, curr_conn_mat = get_source_info_mesh(SOURCE_MODEL_INFO[retrieved_idx[0]]["vertices_mat"], SOURCE_MODEL_INFO[retrieved_idx[0]]["default_param"], SOURCE_MODEL_INFO[retrieved_idx[0]]["constraint_proj_mat"], curr_param.shape[0])

            output_vertices = get_shape_numpy(curr_mat, curr_param, curr_default_param.T, connectivity_mat=curr_conn_mat)
            mesh = trimesh.Trimesh(
                vertices=output_vertices,
                faces=SOURCE_MODEL_INFO[retrieved_idx[0]]["faces"]
            )
        ret_id = SOURCE_MODEL_INFO[retrieved_idx[0]]['model_id'][1]
        if ret_id == model_id:
            exact +=1 
        # mesh.vertices = mesh.vertices + (center - centroid) / np.linalg.norm(diag)
        pred_ind = voxelize_mesh(mesh)
        # gt_mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{model_id}/models/model_normalized.obj", force='mesh')
        # gt_ind = voxelize_mesh(gt_mesh.apply_transform(trimesh.transformations.euler_matrix(0, np.pi/2,0)))

        gt_ind = \
            grids_val[cat_id, model_id]
        
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

        iou = inter / union
        # print(iou)
        ious.append(iou)
    ious = np.asarray(ious)
    print(np.min(ious))
    print(np.max(ious))
    print(np.mean(ious))
    print(np.median(ious))
    print(len(ious[ious > 0.5]))
    print(exact)
    return ious
    # with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/scan2cad_val_top5_ret.json', 'w') as t5f:
    #     json.dump(top5_gt, t5f)

def compare_f1():
    from pytorch3d.io import load_obj
    def get_source_latent_codes_encoder(source_labels, SOURCE_MODEL_INFO, retrieval_encoder, device):
        # print("Using encoder to get source latent codes.")
        source_points = []

        # start_tm = time.time()
        for source_label in source_labels:
            src_points = SOURCE_MODEL_INFO[source_label]["points"]	
            source_points.append(src_points)
        # ret_tm = time.time()
        # print("Load from labels time: ", ret_tm - start_tm)
        # print("Num labels: ", source_labels.shape)
        source_points = np.array(source_points)
        source_points = torch.from_numpy(source_points).to(device, dtype=torch.float)

        src_latent_codes = retrieval_encoder(source_points)
        # print("Retrieval encoding time: ", time.time() - ret_tm)
        return src_latent_codes

    JOINT_BASE_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/'
    JOINT_SRC_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data/roca_sources_centroid/chair/h5'
    JOINT_MODEL_PATH = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/log/chair522_partial/model.pth'
    filename_pickle = JOINT_BASE_DIR+"data/generated_datasplits/chair_522_roca.pickle"
    with open(filename_pickle, 'rb') as handle:
        sources = pickle.load(handle)['sources']
    src_data_fol = JOINT_SRC_DIR
    MAX_NUM_PARAMS = -1
    MAX_NUM_PARTS = -1
    device = 'cuda'
    SOURCE_MODEL_INFO = []
    print("Loading joint sources...")
    with open("/home/karacam/Thesis/joint_learning_retrieval_deformation/shape2part_ext.json", 'r') as f:
        shape2part = json.load(f)
    part2shape = dict([(value, key) for key, value in shape2part['chair']['model_names'].items()])

    for i in range(len(sources)):
        source_model = sources[i]
        src_filename = str(source_model) + "_leaves.h5"

        default_param, points, point_labels, points_mat, \
                    constraint_mat,	constraint_proj_mat	= get_model(src_data_fol + '/' + src_filename, constraint=True)

        curr_source_dict = {}
        curr_source_dict["default_param"] = default_param
        curr_source_dict["points"] = points
        curr_source_dict["point_labels"] = point_labels
        curr_source_dict["points_mat"] = points_mat
        curr_source_dict["model_id"] = (shape2part['chair']['synsetid'], part2shape[str(source_model)])

        curr_source_dict["constraint_mat"] = constraint_mat
        curr_source_dict["constraint_proj_mat"] = constraint_proj_mat
        _, curr_source_dict["vertices_mat"], curr_source_dict["faces"], _ = get_model(os.path.join(src_data_fol, src_filename), pred=True)

        # Get number of parts of the model
        num_parts = len(np.unique(point_labels))
        curr_source_dict["num_parts"] = num_parts

        curr_num_params = default_param.shape[0]
        if (MAX_NUM_PARAMS < curr_num_params):
            MAX_NUM_PARAMS = curr_num_params
            MAX_NUM_PARTS = int(MAX_NUM_PARAMS/6)

        SOURCE_MODEL_INFO.append(curr_source_dict)
    print("Done loading joint sources.")
    embedding_size = 6
    TARGET_LATENT_DIM = 256
    SOURCE_LATENT_DIM = 256
    PART_LATENT_DIM = 32
    target_encoder = TargetEncoder(
        TARGET_LATENT_DIM,
        3,
    )
    decoder_input_dim = TARGET_LATENT_DIM + SOURCE_LATENT_DIM + PART_LATENT_DIM
    param_decoder = ParamDecoder2(decoder_input_dim, 256, embedding_size)
    retrieval_encoder = TargetEncoder(
        TARGET_LATENT_DIM,
        3,
    )

    _model = torch.load(JOINT_MODEL_PATH)
    target_encoder.load_state_dict(_model["target_encoder"])
    target_encoder.to(device)
    target_encoder.eval()

    param_decoder.load_state_dict(_model["param_decoder"])
    param_decoder.to(device)
    param_decoder.eval()

    retrieval_encoder.load_state_dict(_model["retrieval_encoder"])
    retrieval_encoder.to(device)
    retrieval_encoder.eval()

    SOURCE_LATENT_CODES = _model["source_latent_codes"].detach()
    SOURCE_PART_LATENT_CODES = [_x.detach() for _x in _model["part_latent_codes"]]
    SOURCE_VARIANCES = _model["source_variances"].detach()

    source_labels = torch.arange(len(SOURCE_MODEL_INFO))#.repeat(10)
    src_latent_codes = torch.gather(SOURCE_LATENT_CODES, 0, source_labels.to(device).unsqueeze(-1).repeat(1,SOURCE_LATENT_CODES.shape[-1]))

    with torch.no_grad():
        ret_src_latent_codes = []
        num_sets = 20
        interval = int(len(source_labels)/num_sets)

        for j in range(num_sets):
            if (j==num_sets-1):
                curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:], SOURCE_MODEL_INFO, retrieval_encoder, device)
            else:
                curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:(j+1)*interval], SOURCE_MODEL_INFO, retrieval_encoder, device)
            ret_src_latent_codes.append(curr_src_latent_codes)

        ret_src_latent_codes = torch.cat(ret_src_latent_codes)
    
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/points_val.pkl', 'rb') as f:
        points_val = pickle.load(f)
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/val_grids_32.pkl', 'rb') as f:
        grids_val = pickle.load(f)
    assert len({ps['id_cad'] for ps in points_val}) == len(points_val)
    tar_pcs = {(ps['catid_cad'],ps['id_cad']):ps['points'] for ps in points_val}

    exact = 0
    pred_meshes = []
    gt_meshes = []
    for cat_id, model_id in tqdm(tar_pcs):
        # tar_pc_np = mesh.sample(1024)
        tar_pc = tar_pcs[cat_id, model_id]
        # tar_pc = tar_pc[tar_pc[:,0] > 0]
        try:
            with open(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{model_id}/models/model_normalized.json", 'r') as jf:
                _meta = json.load(jf)
            # tar_mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{model_id}/models/model_normalized.obj", force='mesh')
        except FileNotFoundError:
            print("MODEL NOT FOUND!?!?")

        tar_pc = torch.from_numpy(tar_pc).unsqueeze(0).to(device, dtype=torch.float)
        tar_pc.requires_grad = False
        
        retrieval_latent_codes = retrieval_encoder(tar_pc)
        retrieval_latent_codes = retrieval_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
        # retrieval_latent_codes = retrieval_latent_codes.view(-1, retrieval_latent_codes.shape[-1])	
        retrieval_latent_codes = retrieval_latent_codes.view(len(SOURCE_MODEL_INFO), -1, TARGET_LATENT_DIM)
        with torch.no_grad():
            source_labels = source_labels
            _src_latent_codes = ret_src_latent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

            src_variances = get_source_latent_codes_fixed(source_labels, SOURCE_VARIANCES, device=device)
            src_variances = src_variances.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

        distances = compute_mahalanobis(retrieval_latent_codes, _src_latent_codes, src_variances, activation_fn=torch.sigmoid)
        sorted_indices = torch.argsort(distances, dim=0)
        retrieved_idx = sorted_indices[0,:]

        target_latent_codes = target_encoder(tar_pc)
        concat_latent_code = torch.cat((src_latent_codes[retrieved_idx], target_latent_codes), dim=1)

        curr_num_parts = SOURCE_MODEL_INFO[retrieved_idx]["num_parts"]
        curr_code = concat_latent_code[0]
        curr_code_repeated = curr_code.view(1,curr_code.shape[0]).repeat(curr_num_parts, 1)
        
        part_latent_codes = SOURCE_PART_LATENT_CODES[retrieved_idx]

        full_latent_code = torch.cat((curr_code_repeated, part_latent_codes), dim=1)

        params = param_decoder(full_latent_code, use_bn=False)

        ## Pad with extra zero rows to cater to max number of parameters
        if (curr_num_parts < MAX_NUM_PARTS):
            dummy_params = torch.zeros((MAX_NUM_PARTS-curr_num_parts, embedding_size), dtype=torch.float, device=device)
            params = torch.cat((params, dummy_params), dim=0)

        params = params.view(-1, 1)
        pred_params = params.detach().cpu().numpy()
        curr_param = np.expand_dims(pred_params, -1)
        curr_mat, curr_default_param, curr_conn_mat = get_source_info_mesh(SOURCE_MODEL_INFO[retrieved_idx[0]]["vertices_mat"], SOURCE_MODEL_INFO[retrieved_idx[0]]["default_param"], SOURCE_MODEL_INFO[retrieved_idx[0]]["constraint_proj_mat"], curr_param.shape[0])

        output_vertices = get_shape_numpy(curr_mat, curr_param, curr_default_param.T, connectivity_mat=curr_conn_mat)
        mesh = trimesh.Trimesh(
            vertices=output_vertices,
            faces=SOURCE_MODEL_INFO[retrieved_idx[0]]["faces"]
        )
        pred_meshes.append((mesh.vertices, mesh.faces))
        gt_mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{model_id}/models/model_normalized.obj", force='mesh')
        gt_meshes.append((gt_mesh.vertices, gt_mesh.faces))
        ret_id = SOURCE_MODEL_INFO[retrieved_idx[0]]['model_id'][1]
        if ret_id == model_id:
            exact +=1 
    pred_meshes = Meshes([torch.from_numpy(m[0]).type(torch.float) for m in pred_meshes], [torch.from_numpy(m[1]).type(torch.float) for m in pred_meshes])
    gt_meshes = Meshes([torch.from_numpy(m[0]).type(torch.float) for m in gt_meshes], [torch.from_numpy(m[1]).type(torch.float) for m in gt_meshes])
    print("Comparing meshes...")
    metrics = compare_meshes(pred_meshes, gt_meshes)
    print("Done")
    print(metrics)
    # print(len(ious[ious > 0.5]))
    print(exact)
    # with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/scan2cad_val_top5_ret.json', 'w') as t5f:
    #     json.dump(top5_gt, t5f)

def get_all_emds(normalize = False, norm_method = None, sample='fps', num_points=1024):
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/CustomCenter/points_train_center.pkl', 'rb') as f:
        train_objs = list(pickle.load(f))
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/CustomCenter/points_val_center.pkl', 'rb') as f:
        val_objs = list(pickle.load(f))
    
    print("Loading train objs...")
    train_meshes = {}
    train_sampled = {}
    if not normalize and (sample == 'fps'):
        train_sampled = {(a['catid_cad'], a['id_cad']): a['points'] for a in train_objs}
        val_sampled = {(a['catid_cad'], a['id_cad']): a['points'] for a in val_objs}
    else:
        train_objs = [(a['catid_cad'], a['id_cad']) for a in train_objs]
        val_objs = [(a['catid_cad'], a['id_cad']) for a in val_objs]

        for cat_id, model_id in train_objs:
            mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/{cat_id}/{model_id}/models/model_normalized.obj", force='mesh')
            if normalize:
                if norm_method == "iso_norm":
                    m_edge = np.argmax(mesh.bounds[1] - mesh.bounds[0])
                    mesh.vertices = (mesh.vertices - mesh.bounds[0,m_edge]) / (mesh.bounds[1, m_edge] - mesh.bounds[0, m_edge])
                else:
                    mmin, mmax = mesh.bounds
                    mesh.vertices[:,0] = (mesh.vertices[:,0] - mmin[0]) / (mmax[0] - mmin[0])
                    mesh.vertices[:,1] = (mesh.vertices[:,1] - mmin[1]) / (mmax[1] - mmin[1])
                    mesh.vertices[:,2] = (mesh.vertices[:,2] - mmin[2]) / (mmax[2] - mmin[2])
            # Center the mesh
            mesh.apply_translation(-mesh.bounds.sum(axis=0)/2.)
            train_meshes[cat_id, model_id] = mesh
            if sample == 'fps':
                train_sampled[cat_id, model_id] = trimesh.sample.sample_surface_even(mesh, num_points)[0]
            else:
                train_sampled[cat_id, model_id] = mesh.sample(num_points)
        
        print("Loading validation objs...")
        val_meshes = []
        val_sampled = []
        for cat_id, model_id in val_objs:
            mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/{cat_id}/{model_id}/models/model_normalized.obj", force='mesh')
            if normalize:
                if norm_method == "iso_norm":
                    m_edge = np.argmax(mesh.bounds[1] - mesh.bounds[0])
                    mesh.vertices = (mesh.vertices - mesh.bounds[0,m_edge]) / (mesh.bounds[1, m_edge] - mesh.bounds[0, m_edge])
                else:
                    mmin, mmax = mesh.bounds
                    mesh.vertices[:,0] = (mesh.vertices[:,0] - mmin[0]) / (mmax[0] - mmin[0])
                    mesh.vertices[:,1] = (mesh.vertices[:,1] - mmin[1]) / (mmax[1] - mmin[1])
                    mesh.vertices[:,2] = (mesh.vertices[:,2] - mmin[2]) / (mmax[2] - mmin[2])
            mesh.apply_translation(-mesh.bounds.sum(axis=0)/2.)
            val_meshes.append(mesh)
            if sample == 'fps':
                val_sampled.append(trimesh.sample.sample_surface_even(mesh, num_points)[0])
            else:
                val_sampled.append(mesh.sample(num_points))
    
    print("Getting EMD scores...")
    all_emds = {}
    # for id_t, verts_t in tqdm(train_sampled.items()):
    def _cmp(id_t, verts_t):
        try:
            emds_per_t = np.asarray([EMD(verts_t, verts_v) for verts_v in val_sampled])
            return (id_t, emds_per_t)
        except:
            print(id_t, verts_t, val_sampled)
            exit()
    num_cores = int(0.5 * multiprocessing.cpu_count())
    all_emds['scores'] = {}
    [all_emds['scores'].update({_id: _scrs}) for _id, _scrs in Parallel(n_jobs=num_cores)(delayed(_cmp)(id_t, verts_t) for id_t, verts_t in tqdm(train_sampled.items()))]
    all_emds['val_ids'] = val_objs

    print("Dumping results...")
    if normalize:
        if norm_method == "iso_norm":
            with open("all_emds_train2val_iso-norm.pkl", 'wb') as f:
                pickle.dump(all_emds, f)
        else:
            with open("all_emds_train2val_norm.pkl", 'wb') as f:
                pickle.dump(all_emds, f)
    elif sample == 'random':
        with open("all_emds_train2val_random.pkl", 'wb') as f:
            pickle.dump(all_emds, f)
    else:
        with open("all_emds_train2val.pkl", 'wb') as f:
            pickle.dump(all_emds, f)
    print("Done!")
    return

def get_all_metrics(normalize = False, norm_method = None):
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/train_grids_32.pkl', 'rb') as f:
        train_objs = list(pickle.load(f).keys())
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/val_grids_32.pkl', 'rb') as f:
        val_objs = list(pickle.load(f).keys())
    
    print("Loading train objs...")
    train_meshes = {}
    for cat_id, model_id in train_objs:
        mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/{cat_id}/{model_id}/models/model_normalized.obj", force='mesh')
        if normalize:
            if norm_method == "iso_norm":
                m_edge = np.argmax(mesh.bounds[1] - mesh.bounds[0])
                mesh.vertices = (mesh.vertices - mesh.bounds[0,m_edge]) / (mesh.bounds[1, m_edge] - mesh.bounds[0, m_edge])
            else:
                mmin, mmax = mesh.bounds
                mesh.vertices[:,0] = (mesh.vertices[:,0] - mmin[0]) / (mmax[0] - mmin[0])
                mesh.vertices[:,1] = (mesh.vertices[:,1] - mmin[1]) / (mmax[1] - mmin[1])
                mesh.vertices[:,2] = (mesh.vertices[:,2] - mmin[2]) / (mmax[2] - mmin[2])

        mesh.apply_translation(-mesh.bounds.sum(axis=0)/2.)
        train_meshes[cat_id, model_id] = mesh
    
    print("Loading validation objs...")
    val_meshes = []
    for cat_id, model_id in val_objs:
        mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/{cat_id}/{model_id}/models/model_normalized.obj", force='mesh')
        if normalize:
            if norm_method == "iso_norm":
                m_edge = np.argmax(mesh.bounds[1] - mesh.bounds[0])
                mesh.vertices = (mesh.vertices - mesh.bounds[0,m_edge]) / (mesh.bounds[1, m_edge] - mesh.bounds[0, m_edge])
            else:
                mmin, mmax = mesh.bounds
                mesh.vertices[:,0] = (mesh.vertices[:,0] - mmin[0]) / (mmax[0] - mmin[0])
                mesh.vertices[:,1] = (mesh.vertices[:,1] - mmin[1]) / (mmax[1] - mmin[1])
                mesh.vertices[:,2] = (mesh.vertices[:,2] - mmin[2]) / (mmax[2] - mmin[2])
        mesh.apply_translation(-mesh.bounds.sum(axis=0)/2.)
        val_meshes.append(mesh)
    
    print("Getting F1 scores...")
    all_metrics = {'scores':{}}
    for id_t, mesh_t in tqdm(train_meshes.items()):
    # def _cmp(id_t, mesh_t):
        # f1s_per_t = []
        pred_meshes = Meshes([torch.from_numpy(mesh_t.vertices).type(torch.float)]*len(val_meshes),
                             [torch.from_numpy(mesh_t.faces).type(torch.float)]*len(val_meshes)).to('cuda')
        gt_meshes = Meshes([torch.from_numpy(mesh_v.vertices).type(torch.float) for mesh_v in val_meshes],
                           [torch.from_numpy(mesh_v.faces).type(torch.float) for mesh_v in val_meshes]).to('cuda')
        metrics = compare_meshes(pred_meshes, gt_meshes, reduce=False)
        metrics = {k: v.numpy() for k, v in metrics.items()}
        all_metrics['scores'].update({id_t: metrics})
    all_metrics['val_ids'] = val_objs
    print("Dumping results...")
    if normalize:
        if norm_method == "iso_norm":
            with open("all_metrics_train2val_iso-norm.pkl", 'wb') as f:
                pickle.dump(all_metrics, f)
        else:
            with open("all_metrics_train2val_norm.pkl", 'wb') as f:
                pickle.dump(all_metrics, f)
    else:
        with open("all_metrics_train2val.pkl", 'wb') as f:
            pickle.dump(all_metrics, f)
    print("Done!")
    return

def compare_src_ious():
    JOINT_BASE_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/'
    JOINT_SRC_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data/roca_sources_part_thresh_32_1024p_v2/chair/h5'
    filename_pickle = JOINT_BASE_DIR+"data/generated_datasplits/chair_519_roca_v2.pickle"
    with open(filename_pickle, 'rb') as handle:
        sources = pickle.load(handle)['sources']
    src_data_fol = JOINT_SRC_DIR

    with open("/home/karacam/Thesis/joint_learning_retrieval_deformation/shape2part_ext.json", 'r') as f:
        shape2part = json.load(f)
    part2shape = dict([(value, key) for key, value in shape2part['chair']['model_names'].items()])
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/train_grids_32.pkl', 'rb') as f:
        grids_train = pickle.load(f)

    def _cmp(source_model):
        src_filename = str(source_model) + "_leaves.h5"

        with h5py.File(src_data_fol + '/' + src_filename, 'r') as f:
            vertices = f["vertices"][:]
            faces = f["faces"][:]
        model_id = (shape2part['chair']['synsetid'], part2shape[str(source_model)])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        pred_ind = voxelize_mesh(mesh)
        # gt_mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{model_id}/models/model_normalized.obj", force='mesh')
        # gt_ind = voxelize_mesh(gt_mesh.apply_transform(trimesh.transformations.euler_matrix(0, np.pi/2,0)))

        gt_ind = \
            grids_train[model_id]
        
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

        iou = inter / union
        return iou

    num_cores = int(0.4 * multiprocessing.cpu_count())
    ious = Parallel(n_jobs=num_cores)(delayed(_cmp)(source_model) for source_model in tqdm(sources))
    ious = np.asarray(ious)
    print(np.min(ious))
    print(np.max(ious))
    print(np.mean(ious))
    print(np.median(ious))
    print(len(ious[ious > 0.5]))
    return ious
    # with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/scan2cad_val_top5_ret.json', 'w') as t5f:
    #     json.dump(top5_gt, t5f)

def process_top5_gt():
    def get_source_latent_codes_encoder(source_labels, SOURCE_MODEL_INFO, retrieval_encoder, device):
        # print("Using encoder to get source latent codes.")
        source_points = []

        # start_tm = time.time()
        for source_label in source_labels:
            src_points = SOURCE_MODEL_INFO[source_label]["points"]	
            source_points.append(src_points)
        # ret_tm = time.time()
        # print("Load from labels time: ", ret_tm - start_tm)
        # print("Num labels: ", source_labels.shape)
        source_points = np.array(source_points)
        source_points = torch.from_numpy(source_points).to(device, dtype=torch.float)

        src_latent_codes = retrieval_encoder(source_points)
        # print("Retrieval encoding time: ", time.time() - ret_tm)
        return src_latent_codes

    JOINT_BASE_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/'
    JOINT_SRC_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data/roca_sources_centroid/chair/h5'
    JOINT_MODEL_PATH = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/log/chair522_partial/model.pth'
    filename_pickle = JOINT_BASE_DIR+"data/generated_datasplits/chair_522_roca.pickle"
    with open(filename_pickle, 'rb') as handle:
        sources = pickle.load(handle)['sources']
    src_data_fol = JOINT_SRC_DIR

    device = 'cuda'
    SOURCE_MODEL_INFO = []
    print("Loading joint sources...")
    with open("/home/karacam/Thesis/joint_learning_retrieval_deformation/shape2part_ext.json", 'r') as f:
        shape2part = json.load(f)
    part2shape = dict([(value, key) for key, value in shape2part['chair']['model_names'].items()])

    for i in range(len(sources)):
        source_model = sources[i]
        src_filename = str(source_model) + "_leaves.h5"

        default_param, points, point_labels, points_mat, \
                        constraint_mat,	constraint_proj_mat	= get_model(src_data_fol + '/' + src_filename, constraint=True)

        curr_source_dict = {}
        curr_source_dict["default_param"] = default_param
        curr_source_dict["points"] = points
        curr_source_dict["point_labels"] = point_labels
        curr_source_dict["points_mat"] = points_mat
        curr_source_dict["model_id"] = (shape2part['chair']['synsetid'], part2shape[str(source_model)])

        curr_source_dict["constraint_mat"] = constraint_mat
        curr_source_dict["constraint_proj_mat"] = constraint_proj_mat

        # Get number of parts of the model
        num_parts = len(np.unique(point_labels))
        curr_source_dict["num_parts"] = num_parts

        SOURCE_MODEL_INFO.append(curr_source_dict)
    print("Done loading joint sources.")
    TARGET_LATENT_DIM = 256
    SOURCE_LATENT_DIM = 256
    # with torch.no_grad():

    # param_decoder.to(device, dtype=torch.float)
    retrieval_encoder = TargetEncoder(
        TARGET_LATENT_DIM,
        3,
    )
    # embed_loss = nn.TripletMarginLoss(margin=MARGIN)

    _model = torch.load(JOINT_MODEL_PATH)

    retrieval_encoder.load_state_dict(_model["retrieval_encoder"])
    retrieval_encoder.to(device)
    retrieval_encoder.eval()

    SOURCE_VARIANCES = _model["source_variances"].detach()

    source_labels = torch.arange(len(SOURCE_MODEL_INFO))#.repeat(10)


    with torch.no_grad():
        ret_src_latent_codes = []
        num_sets = 20
        interval = int(len(source_labels)/num_sets)

        for j in range(num_sets):
            if (j==num_sets-1):
                curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:], SOURCE_MODEL_INFO, retrieval_encoder, device)
            else:
                curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:(j+1)*interval], SOURCE_MODEL_INFO, retrieval_encoder, device)
            ret_src_latent_codes.append(curr_src_latent_codes)

        ret_src_latent_codes = torch.cat(ret_src_latent_codes)
    
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/points_val.pkl', 'rb') as f:
        points_val = pickle.load(f)
    top5_gt = {s['id_cad']:[] for s in points_val}

    assert len({ps['id_cad'] for ps in points_val}) == len(points_val)
    tar_pcs = {(ps['catid_cad'],ps['id_cad']):ps['points'] for ps in points_val}

    for cat_id, model_id in tqdm(tar_pcs):
        # tar_pc_np = mesh.sample(1024)
        tar_pc = tar_pcs[cat_id, model_id]
        tar_pc = torch.from_numpy(tar_pc).unsqueeze(0).to(device, dtype=torch.float)
        tar_pc.requires_grad = False

        assert tar_pc.shape == (1,1024,3)
        
        retrieval_latent_codes = retrieval_encoder(tar_pc)
        retrieval_latent_codes = retrieval_latent_codes.unsqueeze(0).repeat(len(SOURCE_MODEL_INFO),1,1)
        # retrieval_latent_codes = retrieval_latent_codes.view(-1, retrieval_latent_codes.shape[-1])	
        retrieval_latent_codes = retrieval_latent_codes.view(len(SOURCE_MODEL_INFO), -1, TARGET_LATENT_DIM)
        with torch.no_grad():
            source_labels = source_labels
            _src_latent_codes = ret_src_latent_codes.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

            src_variances = get_source_latent_codes_fixed(source_labels, SOURCE_VARIANCES, device=device)
            src_variances = src_variances.view(len(SOURCE_MODEL_INFO), -1, SOURCE_LATENT_DIM)

        distances = compute_mahalanobis(retrieval_latent_codes, _src_latent_codes, src_variances, activation_fn=torch.sigmoid)
        sorted_indices = torch.argsort(distances, dim=0)
        retrieved_idx = sorted_indices[5,:]
        top5_gt[model_id] = [SOURCE_MODEL_INFO[k]['model_id'][1] for k in retrieved_idx]

    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/scan2cad_val_top5_ret.json', 'w') as t5f:
        json.dump(top5_gt, t5f)

def sample_render_targets():
    with open("/home/karacam/Thesis/joint_learning_retrieval_deformation/shape2part_ext.json", 'r') as f:
        shape2part = json.load(f)
    part2shape = dict([(value, key) for key, value in shape2part['chair']['model_names'].items()])

    target_data_fol = os.path.join("/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data", "roca_targets_part_thresh_32_1024p_v1", "chair", "h5")
    shapenet_root_dir = '/mnt/noraid/karacam/ShapeNetCore.v2/03001627/'
    out_dir = os.path.join("/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data", "roca_targets_partial")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(os.path.join(out_dir, 'chair'))
        os.mkdir(os.path.join(out_dir, 'chair', 'h5'))
    num_cores = int(0.4 * multiprocessing.cpu_count())
    #for tar in tqdm(os.listdir(target_data_fol)):
    def _tmp(fname):
        partid = fname.split('_')[0]
        try:
            shapeid = part2shape[partid]
        except KeyError:
            return
        try:
            mesh = trimesh.load(os.path.join(shapenet_root_dir, shapeid, "models/model_normalized.obj"), force='mesh')
        except ValueError:
            return
        mesh.apply_translation(-mesh.bounds.sum(axis=0)/2.)
        # mesh.apply_transform(trimesh.transformations.euler_matrix(0, np.pi, 0))
        for i in range(4):
            rnd_trs = trimesh.transformations.euler_matrix(np.random.ranf((1,))*np.pi/3, np.random.ranf((1,))*np.pi, 0)
            mesh.apply_transform(rnd_trs)
            pc = render_and_unproject(mesh.vertices, mesh.faces)
            if pc.shape[0] < 1024:
                pc = np.concatenate((pc, np.zeros((1024-pc.shape[0], 3))))
            else:
                pc = np.random.permutation(pc)[:1024]
            pc = trimesh.transform_points(pc, np.linalg.inv(rnd_trs))
            mesh.apply_transform(np.linalg.inv(rnd_trs))
            # trimesh.PointCloud(pc).export('tst_pc.ply')
            # mesh.export('tst_mesh.obj')
            # exit()
            assert (pc.shape == (1024,3))
            out_h5_file = os.path.join(out_dir, 'chair', "h5", str(partid)+'_'+str(i)+'_leaves.h5')
            select_keys = ['vertex_labels', 'face_labels', 'point_labels', 'vertices_mat',
                'points_mat','default_param',
                'point_semantic', 'vertex_semantic']      
            with h5py.File(out_h5_file, 'w') as f:
                f.create_dataset('box_params', data=[], compression="gzip")
                f.create_dataset('orig_ids', data=[partid], compression="gzip")
                f.create_dataset('semantic_label', data=[0], compression="gzip")
                f.create_dataset('points', data=pc, compression="gzip")
                f.create_dataset('vertices', data=mesh.vertices, compression="gzip")
                f.create_dataset('faces', data=mesh.faces, compression="gzip")
                for key in select_keys:
                    f.create_dataset(key, data=[], compression="gzip")
    Parallel(n_jobs=num_cores)(delayed(_tmp)(fname) for fname in tqdm(os.listdir(target_data_fol)))
    # [_tmp(fname) for fname in os.listdir(target_data_fol)]

def get_source_info_mesh(points_mat, default_param, constraint_proj_mat, max_num_params, use_connectivity=True):
    padded_mat = np.zeros((points_mat.shape[0], max_num_params))
    padded_mat[0:points_mat.shape[0], 0:points_mat.shape[1]] = points_mat
    # padded_mat = np.expand_dims(padded_mat, axis=0)

    padded_default_param = np.zeros(max_num_params)
    padded_default_param[:default_param.shape[0]] = default_param
    padded_default_param = np.expand_dims(padded_default_param, axis=0)

    constraint_padded_mat = np.zeros((max_num_params, max_num_params))
    constraint_padded_mat[0:constraint_proj_mat.shape[0], 0:constraint_proj_mat.shape[1]] = constraint_proj_mat	

    return padded_mat.astype("float"), padded_default_param.astype("float"), constraint_padded_mat.astype("float")

def get_shape_numpy(A, param, src_default_param=None, weight=0.1, connectivity_mat = None):
    ### A is the parametric model of the shape
    ### assumes that the shape of A and param agree
    param = np.multiply(param, weight)

    if (src_default_param is None):
        param = param
    else:
        param = param + src_default_param

    # For connectivity
    if connectivity_mat is None:
        param = param

    else:
        # print("Using connectivity constraint for mesh generation.")
        param = np.matmul(connectivity_mat, param)

    pc = np.reshape(np.matmul(A, param), (-1, 3), order='C')

    return pc

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

def EMD(p,q):
    d = cdist(p, q)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(p), len(q))

def render_image_from_mesh(vertices, faces, device='cuda'):
    R, T = look_at_view_transform(3)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras)
    )
    if type(vertices) is not torch.Tensor:
        vertices = torch.from_numpy(vertices).type(torch.float)
    if type(faces) is not torch.Tensor:
        faces = torch.from_numpy(faces).type(torch.float)
    vertices = vertices.to(device)
    faces = faces.to(device)

    mesh = Meshes([vertices], [faces])
    color = torch.ones(1, vertices.size(0), 3, device=device)
    mesh.textures = TexturesVertex(verts_features=color)
    img = renderer(mesh)
    img = (img.detach().cpu().numpy()[0] * 255).astype('uint8')

    return img

def render_and_unproject(vertices, faces, device='cuda'):
    # R, T = look_at_view_transform(1, torch.rand(1)*60, torch.rand(1)*360)
    R, T = look_at_view_transform(1)
    img_size = 256
    cx = cy = fx = fy = img_size / 2.
    cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=((fx,fy),), image_size=((img_size,img_size),), principal_point=((cx,cy),), in_ndc=False)
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(device=device, cameras=cameras)
    )
    if type(vertices) is not torch.Tensor:
        vertices = torch.from_numpy(vertices).type(torch.float)
    if type(faces) is not torch.Tensor:
        faces = torch.from_numpy(faces).type(torch.float)
    vertices = vertices.to(device)
    faces = faces.to(device)

    mesh = Meshes([vertices], [faces])
    color = torch.ones(1, vertices.size(0), 3, device=device)
    mesh.textures = TexturesVertex(verts_features=color)
    img = renderer(mesh)
    fragments = rasterizer(mesh)
    depth = fragments.zbuf.squeeze(-1)
    # _img = (img.detach().cpu().numpy()[0] * 255).astype('uint8')
    # _img = Image.fromarray(_img).convert('RGB')
    # _img.save('tst_render.png')
    # _img.close()

    yy, xx = torch.meshgrid(torch.arange(img.shape[2]-1, -1, -1), torch.arange(img.shape[2]-1, -1, -1))
    xx = xx.to(device)
    yy = yy.to(device)

    xy_depth = torch.stack([xx,yy,depth[0]], dim=-1)
    # _img = Image.fromarray((xy_depth.cpu().numpy()*255).astype('uint8')).convert('RGB')
    # _img.save('tst_depth.png')
    # _img.close()

    xy_depth = xy_depth[xy_depth[:,:,2] > -1]
    pc = cameras.unproject_points(xy_depth)
    # print(torch.cat([pc, torch.ones((pc.shape[0],1)).to(device)], dim=-1) @ TRS[0].inverse())

    pc = pc.cpu().numpy()
    # pc = (pc @ R[0].numpy())

    return pc

def image_grid(imgs, rows, cols):
    # assert len(imgs) == rows*cols

    w, h = imgs[0].size
    dw, dh = (256,256)
    grid = Image.new('RGB', size=(cols*dw, rows*dh))
    
    for i, img in enumerate(imgs):
        im = img.crop(((w-h)//2, 0, w-((w-h)//2), h))
        im = im.resize((dw,dh), Image.ANTIALIAS)
        grid.paste(im, box=(i%cols*dw, i//cols*dh))
    return grid

def get_logs_emd():
    shapenet_root_dir = '/mnt/noraid/karacam/ShapeNetCore.v2/'

    with open("all_emds_train2val.pkl", 'rb') as f:
        all_emds = pickle.load(f)

    val_ids = [(e['catid_cad'], e['id_cad']) for e in all_emds['val_ids']]
    train_ids = list(all_emds['scores'].keys())
    trains_in_val = [id_t for id_t in train_ids if id_t in val_ids]
    trains_not_in_val = [id_t for id_t in train_ids if id_t not in val_ids]
    trains_in_val = np.asarray(trains_in_val)
    trains_not_in_val = np.asarray(trains_not_in_val)
    val_ids = np.asarray(val_ids)

    print(len(trains_in_val), len(trains_not_in_val))
    rnd_trains = np.concatenate((trains_in_val[np.random.choice(len(trains_in_val), 10, replace=False)], 
                    trains_not_in_val[np.random.choice(len(trains_not_in_val), 10, replace=False)]))

    thresh = 0.035

    out_dir = os.path.join('testing', 'emd')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for _ in range(12):
        thresh += 5e-3
        to_grid_emd_imgs = []
        to_grid_f1_imgs = []
        for i, (cat_id, model_id) in enumerate(rnd_trains):
            print(f"Processing {i+1} of {rnd_trains.shape[0]}...")
            mesh_src = trimesh.load(os.path.join(shapenet_root_dir, cat_id, model_id, "models/model_normalized.obj"), force='mesh')
            mesh_src.apply_translation(-mesh_src.bounds.sum(axis=0)/2.)
            mesh_src.apply_transform(trimesh.transformations.euler_matrix(0, np.pi, 0))

            img_src = Image.fromarray(render_image_from_mesh(mesh_src.vertices, mesh_src.faces)).convert('RGB')
            to_grid_emd_imgs.append(img_src)
            to_grid_f1_imgs.append(img_src)

            arr_emd = all_emds['scores'][cat_id, model_id]
            _over_thresh = arr_emd < thresh
            arr_emd = arr_emd[_over_thresh]
            sorted_arr_idx = np.argsort(arr_emd)

            if len(arr_emd) < 10:
                _idx = sorted_arr_idx
            else:
                _idx = sorted_arr_idx[:5]
                _idx = np.concatenate((_idx, sorted_arr_idx[-5:]))
            close_10_dists = arr_emd[_idx]
            # print("F1 scores: ", close_10_dists)
            log = f"> {i+1}: {close_10_dists.tolist()}\n"
            with open(os.path.join(out_dir, f"scores_emd_{thresh:.3f}.log"), 'a') as f:
                f.write(log)

            for cat_id_val, model_id_val in val_ids[_over_thresh][_idx]:
                mesh_val = trimesh.load(os.path.join(shapenet_root_dir, cat_id_val, model_id_val, "models/model_normalized.obj"), force='mesh')
                mesh_val.apply_translation(-mesh_val.bounds.sum(axis=0)/2.)
                mesh_val.apply_transform(trimesh.transformations.euler_matrix(0, np.pi, 0))
                img_val = Image.fromarray(render_image_from_mesh(mesh_val.vertices, mesh_val.faces)).convert('RGB')
                to_grid_emd_imgs.append(img_val)
            if len(arr_emd) < 10:
                for _ in range(10-len(arr_emd)):
                    img_val = Image.fromarray(np.zeros((256,256,3), dtype='uint8')).convert('RGB')
                    to_grid_emd_imgs.append(img_val)
        grid = image_grid(to_grid_emd_imgs, rows=len(rnd_trains), cols=11)
        grid.save(os.path.join(out_dir, f'img_grid_emds_{thresh:.3f}.png'))
        grid.close()
        for img in to_grid_emd_imgs: img.close()

def get_logs_f1(normalize = False, norm_method = None):
    shapenet_root_dir = '/mnt/noraid/karacam/ShapeNetCore.v2/'

    if normalize:
        if norm_method == "iso_norm":
            with open("all_metrics_train2val_iso-norm.pkl", 'rb') as f:
                all_metrics = pickle.load(f)
        else:
            with open("all_metrics_train2val_norm.pkl", 'rb') as f:
                all_metrics = pickle.load(f)
    else:
        with open("all_metrics_train2val.pkl", 'rb') as f:
            all_metrics = pickle.load(f)

    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/val_grids_32.pkl', 'rb') as f:
        grids_val = pickle.load(f)
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/train_grids_32.pkl', 'rb') as f:
        grids_train = pickle.load(f)

    train_ids = list(all_metrics['scores'].keys())
    trains_in_val = [id_t for id_t in train_ids if id_t in all_metrics['val_ids']]
    trains_not_in_val = [id_t for id_t in train_ids if id_t not in all_metrics['val_ids']]
    trains_in_val = np.asarray(trains_in_val)
    trains_not_in_val = np.asarray(trains_not_in_val)

    val_ids = np.asarray(all_metrics['val_ids'])
    assert (val_ids == np.asarray(all_metrics['val_ids'])).all()

    rnd_trains = np.concatenate((trains_in_val[np.random.choice(len(trains_in_val), 10, replace=False)], 
                    trains_not_in_val[np.random.choice(len(trains_not_in_val), 10, replace=False)]))

    f1s_thresh = {
        0.1 : [5, 10, 15, 20, 50, 75], 
        0.3 : [20, 40, 60, 80], 
        0.5 : [20, 40, 60, 80],
    }

    out_dir = 'testing'
    if normalize:
        if norm_method == "iso_norm":
            out_dir = os.path.join(out_dir, 'iso_norm')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
        else:
            out_dir = os.path.join(out_dir, 'norm')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)


    for _f1t, _f1hs in f1s_thresh.items():
        print(f"*** F1 Threshold: {_f1t} ***")
        for _f1h in _f1hs:
            print(f"*** F1 Hyper: {_f1h} ***")
            to_grid_f1_imgs = []
            for i, (cat_id, model_id) in enumerate(rnd_trains):
                print(f"Processing {i+1} of {rnd_trains.shape[0]}...")
                mesh_src = trimesh.load(os.path.join(shapenet_root_dir, cat_id, model_id, "models/model_normalized.obj"), force='mesh')
                if normalize:
                    if norm_method == "iso_norm":
                        m_edge = np.argmax(mesh_src.bounds[1] - mesh_src.bounds[0])
                        mesh_src.vertices = (mesh_src.vertices - mesh_src.bounds[0,m_edge]) / (mesh_src.bounds[1, m_edge] - mesh_src.bounds[0, m_edge])
                    else:
                        mmin, mmax = mesh_src.bounds
                        mesh_src.vertices[:,0] = (mesh_src.vertices[:,0] - mmin[0]) / (mmax[0] - mmin[0])
                        mesh_src.vertices[:,1] = (mesh_src.vertices[:,1] - mmin[1]) / (mmax[1] - mmin[1])
                        mesh_src.vertices[:,2] = (mesh_src.vertices[:,2] - mmin[2]) / (mmax[2] - mmin[2])
                mesh_src.apply_translation(-mesh_src.bounds.sum(axis=0)/2.)
                mesh_src.apply_transform(trimesh.transformations.euler_matrix(0, np.pi, 0))

                img_src = Image.fromarray(render_image_from_mesh(mesh_src.vertices, mesh_src.faces)).convert('RGB')
                to_grid_f1_imgs.append(img_src)

                arr_f1 = all_metrics['scores'][cat_id, model_id][f"F1@{_f1t:.6f}"]
                _over_thresh = arr_f1 > _f1h
                arr_f1 = arr_f1[_over_thresh]
                sorted_arr_idx = np.argsort(arr_f1)
                if len(arr_f1) < 10:
                    _idx = sorted_arr_idx
                else:
                    _idx = sorted_arr_idx[-5:][::-1]
                    _idx = np.concatenate((_idx, sorted_arr_idx[:5][::-1]))
                close_10_dists = arr_f1[_idx]
                log = f"> {i+1}: {close_10_dists.tolist()}\n"
                with open(os.path.join(out_dir, f"scores_{_f1t}_{_f1h}.log"), 'a') as f:
                    f.write(log)

                for cat_id_val, model_id_val in val_ids[_over_thresh][_idx]:
                    mesh_val = trimesh.load(os.path.join(shapenet_root_dir, cat_id_val, model_id_val, "models/model_normalized.obj"), force='mesh')
                    if normalize:
                        if norm_method == "iso_norm":
                            m_edge = np.argmax(mesh_val.bounds[1] - mesh_val.bounds[0])
                            mesh_val.vertices = (mesh_val.vertices - mesh_val.bounds[0,m_edge]) / (mesh_val.bounds[1, m_edge] - mesh_val.bounds[0, m_edge])
                        else:
                            mmin, mmax = mesh_val.bounds
                            mesh_val.vertices[:,0] = (mesh_val.vertices[:,0] - mmin[0]) / (mmax[0] - mmin[0])
                            mesh_val.vertices[:,1] = (mesh_val.vertices[:,1] - mmin[1]) / (mmax[1] - mmin[1])
                            mesh_val.vertices[:,2] = (mesh_val.vertices[:,2] - mmin[2]) / (mmax[2] - mmin[2])
                    mesh_val.apply_translation(-mesh_val.bounds.sum(axis=0)/2.)
                    mesh_val.apply_transform(trimesh.transformations.euler_matrix(0, np.pi, 0))
                    img_val = Image.fromarray(render_image_from_mesh(mesh_val.vertices, mesh_val.faces)).convert('RGB')
                    to_grid_f1_imgs.append(img_val)
                if len(arr_f1) < 10:
                    for _ in range(10-len(arr_f1)):
                        img_val = Image.fromarray(np.zeros((256,256,3), dtype='uint8')).convert('RGB')
                        to_grid_f1_imgs.append(img_val)
            grid = image_grid(to_grid_f1_imgs, rows=len(rnd_trains), cols=11)
            grid.save(os.path.join(out_dir, f'img_grid_f1s_{_f1t}_{_f1h}.png'))
            grid.close()
            for img in to_grid_f1_imgs: img.close()

def process_centered_full_annots():
    shapenet_root_dir = '/mnt/noraid/karacam/ShapeNetCore.v2/'
    with open("/mnt/noraid/karacam/Roca/Data/full_annotations.json", 'r') as f:
        annots = json.load(f)
    # def _tmp(annot):
    for annot in tqdm(annots):
        for m in annot['aligned_models']:
            t = m['trs']['translation']
            q = m['trs']['rotation']
            s = m['trs']['scale']
            mesh = trimesh.load(os.path.join(shapenet_root_dir, m['catid_cad'], m['id_cad'], "models/model_normalized.obj"), force='mesh')
            C = -mesh.bounds.sum(axis=0)/2.
            I = np.eye(4)
            I[:3, 3] -= C
            m['trs']['translation'], m['trs']['rotation'], m['trs']['scale'] = map(list, decompose_mat4(make_M_from_tqs(t,q,s) @ I))
            mesh.apply_translation(C)
            m['center'] = (mesh.bounds.sum(axis=0)/2.).tolist()
        # return annot
    # num_cores = int(0.4 * multiprocessing.cpu_count())
    # new_annots = [Parallel(n_jobs=num_cores)(delayed(_tmp)(annot) for annot in tqdm(annots))]
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/CustomCenter/full_annotations_centered.json", 'w') as f:
        json.dump(annots, f)

def make_shape2part():
    with open("/mnt/noraid/karacam/ThesisData/data/shapenetcore.taxonomy.json", 'r') as json_file:
        shapenet_taxonomy = json.load(json_file)

    all_ids = {}
    for data in shapenet_taxonomy:
        data["level"] = 0
        queue = [data]
        # leaves = []

        while queue:
            node = queue.pop()

            if 'children' not in node:
                all_ids[node['li_attr']['id']] = node['metadata']['label']
            else:
                for child in node['children']:
                    child["level"] = node["level"] + 1
                    child['metadata']["label"] = node['metadata']["label"] + "/" + child['metadata']["label"]
                    all_ids[node['li_attr']['id']] = node['metadata']['label']
                    queue.append(child)
    obj_meta_paths = glob("/mnt/noraid/karacam/data_v0/*/meta.json")
    shape2part = {}
    for p in obj_meta_paths:
        with open(p) as meta_file:
            m = json.load(meta_file)
            shape2part[m["model_id"]] = (m["anno_id"], m['model_cat'].lower())
    with open("shape2part.json", 'w') as f:
        json.dump(shape2part, f)

def make_shape2part_ext():
    CAD_TAXONOMY = {
        '02747177': 'bin',
        '02808440': 'bathtub',
        '02818832': 'bed',
        '02871439': 'bookcase',
        '02933112': 'cabinet',
        '03001627': 'chair',
        '03211117': 'display',
        '04256520': 'sofa',
        '04379243': 'table'
    }
    CAD_TAXONOMY_REVERSE = {v: k for k, v in CAD_TAXONOMY.items()}
    obj_meta_paths = glob("/mnt/noraid/karacam/data_v0/*/meta.json")
    shape2part = {}
    for p in obj_meta_paths:
        with open(p) as meta_file:
            m = json.load(meta_file)
            m_name = m['model_cat'].lower()
            if m_name == 'trashcan':
                m_name = 'bin'
            elif m_name == 'storagefurniture':
                # This is hacky!
                m_name = 'cabinet'
            elif m_name not in CAD_TAXONOMY_REVERSE:
                continue
            if not m_name in shape2part:
                shape2part[m_name] = {'model_names':{}, 'synsetid': CAD_TAXONOMY_REVERSE[m_name]}
            shape2part[m_name]['model_names'][m["model_id"]] = m["anno_id"]
    for k in shape2part:
        shape2part[k]['num_samples'] = len(shape2part[k]['model_names'])
    with open("shape2part_ext.json", 'w') as f:
        json.dump(shape2part, f)

def center_instances():
    shapenet_root_dir = '/mnt/noraid/karacam/ShapeNetCore.v2/'
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_instances_train.json", 'r') as f:
        instances = json.load(f)

    for annot in tqdm(instances['annotations']):
        mesh = trimesh.load(os.path.join(shapenet_root_dir, annot['model']['catid_cad'], annot['model']['id_cad'], "models/model_normalized.obj"), force='mesh')
        t = annot['t']
        q = annot['q']
        s = annot['s']

        C = mesh.bounds.sum(axis=0)/2.
        I = np.eye(4)
        I[:3, 3] += C
        annot['t'], annot['q'], annot['s'] = decompose_mat4(make_M_from_tqs(t, q, s) @ I)
        annot['t'], annot['q'], annot['s'] = (annot['t'].tolist(), annot['q'].tolist(), annot['s'].tolist())

    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_instances_train.json", 'w') as f:
        json.dump(instances, f)

    with open("/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_instances_val.json", 'r') as f:
        instances = json.load(f)

    for annot in tqdm(instances['annotations']):
        mesh = trimesh.load(os.path.join(shapenet_root_dir, annot['model']['catid_cad'], annot['model']['id_cad'], "models/model_normalized.obj"), force='mesh')
        t = annot['t']
        q = annot['q']
        s = annot['s']

        C = mesh.bounds.sum(axis=0)/2.
        I = np.eye(4)
        I[:3, 3] += C
        annot['t'], annot['q'], annot['s'] = decompose_mat4(make_M_from_tqs(t, q, s) @ I)
        annot['t'], annot['q'], annot['s'] = (annot['t'].tolist(), annot['q'].tolist(), annot['s'].tolist())

    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_instances_val.json", 'w') as f:
        json.dump(instances, f)

    with open("/mnt/noraid/karacam/ThesisData/Dataset/scan2cad_val_scenes.json", "r") as f:
        val_scenes = json.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_val_cads.pkl", 'rb') as f:
        val_cads = pickle.load(f)
    val_cads_dict = {c['id_cad']:c for c in val_cads}

    mc = 0
    for scene in val_scenes:
        for inst in val_scenes[scene]:
            if inst['id_cad'] not in val_cads_dict:
                mc+=1
                continue
            mesh = trimesh.Trimesh(
                vertices=val_cads_dict[inst['id_cad']]['verts'], 
                faces=val_cads_dict[inst['id_cad']]['faces']
            )
            C = mesh.bounds.sum(axis=0)/2.
            I = np.eye(4)
            I[:3, 3] += C
            inst['t'], inst['q'], inst['s'] = decompose_mat4(make_M_from_tqs(inst['t'], inst['q'], inst['s']) @ I)
            inst['t'], inst['q'], inst['s'] = (inst['t'].tolist(), inst['q'].tolist(), inst['s'].tolist())

    # Save new annotations
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_val_scenes.json", 'w') as f:
        json.dump(val_scenes, f)

def center_points_and_grids():
    cad_path = '/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_{}_cads.pkl'
    with open(cad_path.format('train'), 'rb') as f:
        train_cads = pickle.load(f)
    with open(cad_path.format('val'), 'rb') as f:
        val_cads = pickle.load(f)

    grids_train = {}
    grids_val = {}
    points_train = []
    points_val = []
    points_extra_keys = ['catid_cad', 'id_cad', 'category_id']

    for pd in tqdm(train_cads):
        # mesh = trimesh.load(os.path.join(shapenet_root_dir, pd['catid_cad'], pd['id_cad'], "models/model_normalized.obj"), force='mesh')
        mesh = trimesh.Trimesh(vertices=pd['verts'], faces=pd['faces'])
        mesh.apply_translation(-mesh.bounds.sum(axis=0)/2.)

        samples = mesh.sample(10000)
        pc,_ = sample_farthest_points(torch.from_numpy(samples[None,...]).to('cuda'), K=1024)
        pcd = {'points': pc.cpu().numpy()[0]}
        for k in points_extra_keys:
            pcd[k] = pd[k]
        points_train.append(pcd)

        grid = voxelize_mesh(mesh)
        grids_train[pd['catid_cad'], pd['id_cad']] = grid
        pd['verts'] = pd['verts'] - mesh.bounds.sum(axis=0)/2.

    for pd in tqdm(val_cads):
        # mesh = trimesh.load(os.path.join(shapenet_root_dir, pd['catid_cad'], pd['id_cad'], "models/model_normalized.obj"), force='mesh')
        mesh = trimesh.Trimesh(vertices=pd['verts'], faces=pd['faces'])
        mesh.apply_translation(-mesh.bounds.sum(axis=0)/2.)
        
        samples = mesh.sample(10000)
        pc,_ = sample_farthest_points(torch.from_numpy(samples[None,...]).to('cuda'), K=1024)
        pcd = {'points': pc.cpu().numpy()}
        for k in points_extra_keys:
            pcd[k] = pd[k]
        points_val.append(pcd)
        
        grid = voxelize_mesh(mesh)
        grids_val[pd['catid_cad'], pd['id_cad']] = grid
        pd['verts'] = pd['verts'] - mesh.bounds.sum(axis=0)/2.

    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Centered/points_train.pkl', 'wb') as f:
        pickle.dump(points_train, f)
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Centered/points_val.pkl', 'wb') as f:
        pickle.dump(points_val, f)
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Centered/train_grids_32.pkl', 'wb') as f:
        pickle.dump(grids_train, f)
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Centered/val_grids_32.pkl', 'wb') as f:
        pickle.dump(grids_val, f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_train_cads.pkl", 'wb') as f:
        pickle.dump(train_cads, f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_val_cads.pkl", 'wb') as f:
        pickle.dump(val_cads, f)

def make_multiclass_instances(num_classes=3):
    assert num_classes in [3,5]
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_instances_train.json", 'r') as f:
        instances_train = json.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_instances_val.json", 'r') as f:
        instances_val = json.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/points_train.pkl", 'rb') as f:
        points_train = pickle.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/points_val.pkl", 'rb') as f:
        points_val = pickle.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/train_grids_32.pkl", 'rb') as f:
        grids_train = pickle.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/val_grids_32.pkl", 'rb') as f:
        grids_val = pickle.load(f)
    with open("/mnt/noraid/karacam/ThesisData/data/roca_storagefurniture.json", 'r') as f:
        storagefurniture_ids = json.load(f)
        storagefurniture_ids = set([_e[1] for _e in storagefurniture_ids])
    
    new_categories = [{'name':'chair', 'id':0}, {'name':'table', 'id':1}, {'name':'storagefurniture', 'id':2}]
    if num_classes == 5:
        new_categories.append({'name':'display', 'id':3})
        new_categories.append({'name':'bin', 'id':4})
    cat_names_ids = {d['name']:d['id'] for d in new_categories}

    def prune_instances(instances, points, grids, storagefurniture_ids):
        new_instances = {'categories':[], 'annotations':[], 'images':[]}
        categories = {d['id']:d['name'] for d in instances['categories']}
        new_img_ids = set({})
        new_annots = []
        all_ids = set({})

        for annot in instances['annotations']:
            cat_name = categories[annot['category_id']]
            if (cat_name in cat_names_ids) or (annot['model']['id_cad'] in storagefurniture_ids):
                if annot['model']['id_cad'] in storagefurniture_ids:
                    annot['category_id'] = 2
                else:
                    annot['category_id'] = cat_names_ids[cat_name]
                new_annots.append(annot)
                new_img_ids.add(annot['image_id'])
                all_ids.add(annot['model']['id_cad'])
        new_imgs = [_img for _img in instances['images'] if _img['id'] in new_img_ids]
        new_instances['categories'] = new_categories
        new_instances['annotations'] = new_annots
        new_instances['images'] = new_imgs

        new_points = [pcd for pcd in points if pcd['id_cad'] in all_ids]
        for pcd in new_points:
            cat_name = categories[pcd['category_id']]
            if pcd['id_cad'] in storagefurniture_ids:
                pcd['category_id'] = 2
            elif cat_name in cat_names_ids:
                pcd['category_id'] = cat_names_ids[cat_name]
        new_grids = {(catid_cad, id_cad):_g for (catid_cad, id_cad), _g in grids.items() if id_cad in all_ids}
        return new_instances, new_points, new_grids

    print("Pruning train...")
    new_instances_train, new_points_train, new_grids_train = prune_instances(instances_train, points_train, grids_train, storagefurniture_ids)
    print("Pruning val...")
    new_instances_val, new_points_val, new_grids_val = prune_instances(instances_val, points_val, grids_val, storagefurniture_ids)
    print("New instances sizes:")
    print(len(new_instances_train['annotations']), len(new_instances_val['annotations']))
    print(len(new_instances_train['images']), len(new_instances_val['images']))
    print(len(new_points_train), len(new_points_val))
    print(len(new_grids_train), len(new_grids_val))
    print("Old instances sizes:")
    print(len(instances_train['annotations']), len(instances_val['annotations']))
    print(len(instances_train['images']), len(instances_val['images']))
    print(len(points_train), len(points_val))
    print(len(grids_train), len(grids_val))
    print()
    print("Saving...")
    if not os.path.exists(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class"):
        os.mkdir(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class")
    # Save new instances
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/scan2cad_instances_train.json", 'w') as f:
        json.dump(new_instances_train, f)
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/scan2cad_instances_val.json", 'w') as f:
        json.dump(new_instances_val, f)
    # Save new points
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/points_train.pkl", 'wb') as f:
        pickle.dump(new_points_train, f)
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/points_val.pkl", 'wb') as f:
        pickle.dump(new_points_val, f)
    # Save new grids
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/train_grids_32.pkl", 'wb') as f:
        pickle.dump(new_grids_train, f)
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/val_grids_32.pkl", 'wb') as f:
        pickle.dump(new_grids_val, f)

    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/scan2cad_instances_train.json", 'r') as f:
        instances_train = json.load(f)
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/scan2cad_instances_val.json", 'r') as f:
        instances_val = json.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/CustomCenter/full_annotations_centered.json", 'r') as f:
        full_annots_centered = json.load(f)

    print("Pruning full annotations...")
    all_ids = set([a['model']['id_cad'] for a in instances_train['annotations']])
    all_ids.update([a['model']['id_cad'] for a in instances_val['annotations']])
    new_full_annots = []
    for a in tqdm(full_annots_centered):
        new_aligned_models = []
        for m in a['aligned_models']:
            if m['id_cad'] in all_ids:
                new_aligned_models.append(m)
        if len(new_aligned_models) > 0:
            a['aligned_models'] = new_aligned_models
            a['n_aligned_models'] = len(new_aligned_models)
            new_full_annots.append(a)
    
    print("New full annotations size:")
    print(len(new_full_annots))
    print("Old full annotations size:")
    print(len(full_annots_centered))

    print("Saving...")
    # Save new annotations
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/full_annotations_centered.json", 'w') as f:
        json.dump(new_full_annots, f)
    
    print("Pruning cads...")
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_train_cads.pkl", 'rb') as f:
        train_cads = pickle.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_val_cads.pkl", 'rb') as f:
        val_cads = pickle.load(f)
    new_train_cads = [c for c in train_cads if c['id_cad'] in all_ids]
    new_val_cads = [c for c in val_cads if c['id_cad'] in all_ids]
    print("Saving new cads...")
    # Save new cads
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/scan2cad_train_cads.pkl", 'wb') as f:
        pickle.dump(new_train_cads, f)
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/scan2cad_val_cads.pkl", 'wb') as f:
        pickle.dump(new_val_cads, f)

    print("Pruning val scenes...")
    # Process val scenes
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/scan2cad_instances_val.json", 'r') as f:
        instances_val = json.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_instances_val.json", 'r') as f:
        instances_val_centered = json.load(f)
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Centered/scan2cad_val_scenes.json", 'r') as f:
        val_scenes = json.load(f)

    categories = {d['id']:d['name'] for d in instances_val_centered['categories']}
    val_ids = set([a['model']['id_cad'] for a in instances_val['annotations']])
    new_val_scenes = {}
    for scene in val_scenes:
        new_val_annots = []
        for inst in val_scenes[scene]:
            cat_name = categories[inst['category_id']]
            if inst['id_cad'] in val_ids:
                if inst['id_cad'] in storagefurniture_ids:
                    inst['category_id'] = 2
                else:
                    inst['category_id'] = cat_names_ids[cat_name]
                new_val_annots.append(inst)
        if len(new_val_annots) > 0:
            new_val_scenes[scene] = new_val_annots
    print("Saving new val scenes...")
    with open(f"/mnt/noraid/karacam/Roca/Data/Dataset/Custom{num_classes}Class/scan2cad_val_scenes.json", 'w') as f:
        json.dump(new_val_scenes, f)
    print("Done!")

def duplicate_images_3class():
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Custom3Class/scan2cad_instances_train.json", 'r') as f:
        instances_train = json.load(f)
    chair_img_ids = set([a['image_id'] for a in instances_train['annotations'] if a['category_id'] == 0])
    table_img_ids = set([a['image_id'] for a in instances_train['annotations'] if a['category_id'] == 1])
    storagefurniture_img_ids = set([a['image_id'] for a in instances_train['annotations'] if a['category_id'] == 2])
    # Get the difference between table and chair
    table_chair_diff = table_img_ids.difference(chair_img_ids)
    # Get the difference between storagefurniture and chair
    storagefurniture_chair_diff = storagefurniture_img_ids.difference(chair_img_ids)
    imgs_dict = {img['id']:img for img in instances_train['images']}

    # Duplicate the difference between table and chair in instances images and annotations, and assign new image ids
    max_img_id = max(imgs_dict.keys())
    max_annot_id = max([a['id'] for a in instances_train['annotations']])
    for img_id in tqdm(table_chair_diff):
        img = imgs_dict[img_id]
        new_img = img.copy()
        max_img_id += 1
        new_img['id'] = max_img_id
        instances_train['images'].append(new_img)
        for annot in instances_train['annotations']:
            if annot['image_id'] == img_id:
                new_annot = annot.copy()
                new_annot['image_id'] = new_img['id']
                max_annot_id += 1
                new_annot['id'] = max_annot_id
                instances_train['annotations'].append(new_annot)

    # Duplicate twice the difference between storagefurniture and chair in instances images and annotations, and assign new image ids
    for img_id in tqdm(storagefurniture_chair_diff):
        img = imgs_dict[img_id]
        new_img = img.copy()
        max_img_id += 1
        new_img['id'] = max_img_id
        instances_train['images'].append(new_img)
        for annot in instances_train['annotations']:
            if annot['image_id'] == img_id:
                new_annot = annot.copy()
                new_annot['image_id'] = new_img['id']
                max_annot_id += 1
                new_annot['id'] = max_annot_id
                instances_train['annotations'].append(new_annot)
        new_img = img.copy()
        max_img_id += 1
        new_img['id'] = max_img_id
        instances_train['images'].append(new_img)
        for annot in instances_train['annotations']:
            if annot['image_id'] == img_id:
                new_annot = annot.copy()
                new_annot['image_id'] = new_img['id']
                max_annot_id += 1
                new_annot['id'] = max_annot_id
                instances_train['annotations'].append(new_annot)

    # Save new instances
    with open("/mnt/noraid/karacam/Roca/Data/Dataset/Custom3Class/scan2cad_instances_train_dup.json", 'w') as f:
        json.dump(instances_train, f)

# with torch.no_grad():
#     ious = compare_ious()
# with torch.no_grad():
#     compare_f1()
# compare_src_ious()
# proccess_annotations()
# process_400k_custom(split='val')
# process_centered_full_annots()
# joint_input_data(split='val')
# make_mock_instances(split='val')
# process_top5_gt()
# get_all_metrics(normalize=True)
# get_logs_f1(normalize=True, norm_method='iso_norm')
# center_instances()
# center_points_and_grids()
# make_multiclass_instances(num_classes=5)
# make_shape2part_ext()

# with open("/home/karacam/Thesis/joint_learning_retrieval_deformation/shape2part_ext.json", 'r') as f:
#     shape2part = json.load(f)
# part2shape = dict([(value, key) for key, value in shape2part['chair']['model_names'].items()])
# trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{part2shape['37698']}/models/model_normalized.obj", force='mesh').export('tst_tar.obj')
# trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v2/03001627/{part2shape['2267']}/models/model_normalized.obj", force='mesh').export('tst_src.obj')

""" 
Storagefurniture dataset 
"""

# with open("shape2part.json", 'r') as json_file:
#     shape2part = json.load(json_file)
# with open("/mnt/noraid/karacam/ThesisData/data/RocaDataset/scan2cad_train_cads.pkl", 'rb') as f:
#     train_cads = pickle.load(f)

# roca_all = {}
# for cad in train_cads:
#     if cad['id_cad'] not in shape2part:
#         continue
#     cat_name = shape2part[cad['id_cad']][1]
#     if cat_name not in roca_all:
#         roca_all[cat_name] = []
#     roca_all[cat_name].append((cad['catid_cad'], cad['id_cad']))
# with open("metadata/scan2cad_taxonomy_9.json", 'r') as f:
#     scan2cad_taxonomy = json.load(f)

# to_remove=[]
# synset_names = {c['shapenet']:c['name'] for c in scan2cad_taxonomy}
# for i, (catid_cad, id_cad) in enumerate(roca_all['storagefurniture']):
#     if synset_names[catid_cad] == 'table' or synset_names[catid_cad] == 'bathtub':
#         to_remove.append(i)
# for i in to_remove[::-1]:
#     roca_all['storagefurniture'].pop(i)
# with open("roca_storagefurniture.json", 'w') as f:
#     json.dump(roca_all['storagefurniture'], f)
"""--------------------------------------------------------"""
img_fname = "tasks/scannet_frames_25k/scene0461_00/color/000200.jpg"

img = Image.open(os.path.join("/mnt/noraid/karacam/Roca/Data/Images", img_fname))
with open("/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_instances_val.json", 'r') as f:
    instances = json.load(f)
imgs = {img['id']: img['file_name'] for img in instances['images']}
for inst_img in instances['images']:
    if inst_img['file_name'] == img_fname:
        img_id = inst_img['id']
        break

img_annots = [a for a in instances['annotations'] if a['image_id'] == img_id]
print([(e['model']['catid_cad'], e['model']['id_cad']) for e in img_annots])
shapenet_root_dir = '/mnt/noraid/karacam/ShapeNetCore.v2/'
tar_mesh = trimesh.load(os.path.join(shapenet_root_dir, '04256520', 'e9e5da988215f06513292732a7b1ed9a', "models/model_normalized.obj"), force='mesh').export('tst_tar.obj')

print("Loading joint sources...")
MAX_NUM_PARAMS = -1
MAX_NUM_PARTS = -1
ALPHA = 0.1
USE_SYMMETRY = True
K = 10
MARGIN = .5
SOURCE_MODEL_INFO = []
filename_pickle = "/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data/generated_datasplits/chair_522_roca.pickle"
src_data_fol = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data/roca_sources_part_thresh_32_1024p_v1/chair/h5/'
with open(filename_pickle, 'rb') as handle:
    sources = pickle.load(handle)['sources']
# with open("/home/karacam/Thesis/joint_learning_retrieval_deformation/shape2part_ext.json", 'r') as f:
with open("/home/karacam/Thesis/ROCA/shape2part_ext.json", 'r') as f:
    shape2part = json.load(f)

part2shape = dict([(value, key) for k in shape2part for key, value in shape2part[k]['model_names'].items()])

for i in range(len(sources)):
    source_model = sources[i]
    src_filename = str(source_model) + "_leaves.h5"

    box_params, orig_ids, default_param, points, point_labels, points_mat, \
        vertices, vertices_mat, faces, face_labels, \
        constraint_mat,	constraint_proj_mat	= get_model(os.path.join(src_data_fol, src_filename), mesh=True, constraint=True)

    curr_source_dict = {}
    curr_source_dict["default_param"] = default_param
    curr_source_dict["points"] = points
    curr_source_dict["point_labels"] = point_labels
    curr_source_dict["points_mat"] = points_mat
    curr_source_dict["model_id"] = (shape2part['chair']['synsetid'], part2shape[str(source_model)])
    curr_source_dict["vertices"] = vertices
    curr_source_dict["vertices_mat"] = vertices_mat
    curr_source_dict["faces"] = faces
    curr_source_dict["face_labels"] = face_labels
    curr_source_dict["constraint_mat"] = constraint_mat
    curr_source_dict["constraint_proj_mat"] = constraint_proj_mat

    # Get number of parts of the model
    num_parts = len(np.unique(point_labels))
    curr_source_dict["num_parts"] = num_parts

    curr_num_params = default_param.shape[0]
    if (MAX_NUM_PARAMS < curr_num_params):
        MAX_NUM_PARAMS = curr_num_params
        MAX_NUM_PARTS = int(MAX_NUM_PARAMS/6)

    SOURCE_MODEL_INFO.append(curr_source_dict)
print("Done loading joint sources.")

embedding_size = 6
TARGET_LATENT_DIM = 256
SOURCE_LATENT_DIM = 256
PART_LATENT_DIM = 32
#### Load model
target_encoder = TargetEncoder(
    256,
    3,
)
device = 'cuda'
target_encoder.to(device, dtype=torch.float)

decoder_input_dim = TARGET_LATENT_DIM + SOURCE_LATENT_DIM + PART_LATENT_DIM
param_decoder = ParamDecoder2(decoder_input_dim, 256, embedding_size)
param_decoder.to(device, dtype=torch.float)

## For Retrieval
retrieval_encoder = TargetEncoder(
    TARGET_LATENT_DIM,
    3,
)
retrieval_encoder.to(device, dtype=torch.float)	

fname = os.path.join("/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/log/chair522_icp/model.pth")
target_encoder.load_state_dict(torch.load(fname)["target_encoder"])
target_encoder.to(device)
target_encoder.eval()

param_decoder.load_state_dict(torch.load(fname)["param_decoder"])
param_decoder.to(device)
param_decoder.eval()

SOURCE_LATENT_CODES = torch.load(fname)["source_latent_codes"]
SOURCE_PART_LATENT_CODES = torch.load(fname)["part_latent_codes"]

retrieval_encoder.load_state_dict(torch.load(fname)["retrieval_encoder"])
retrieval_encoder.to(device)
retrieval_encoder.eval()
SOURCE_VARIANCES = torch.load(fname)["source_variances"]

source_labels = torch.arange(len(SOURCE_MODEL_INFO))#.repeat(10)
src_mats, src_default_params, src_connectivity_mat = get_source_info(source_labels, SOURCE_MODEL_INFO, MAX_NUM_PARAMS, use_connectivity=True)
src_latent_codes = torch.gather(SOURCE_LATENT_CODES, 0, source_labels.to(device).unsqueeze(-1).repeat(1,SOURCE_LATENT_CODES.shape[-1]))

mat = torch.stack([mat for mat in src_mats])#.to(device, dtype=torch.float)
def_param = torch.stack([def_param for def_param in src_default_params])#.to(device, dtype=torch.float)
conn_mat = torch.stack([conn_mat for conn_mat in src_connectivity_mat])
def get_source_latent_codes_encoder(self, source_labels):
    source_points = []
    source_points = np.array([SOURCE_MODEL_INFO[source_label]["points"] for source_label in source_labels])
    source_points = torch.from_numpy(source_points).to(device, dtype=torch.float)

    src_latent_codes = retrieval_encoder(source_points)
    return src_latent_codes
with torch.no_grad():
    ret_src_latent_codes = []
    num_sets = 20
    interval = int(len(source_labels)/num_sets)

    for j in range(num_sets):
        if (j==num_sets-1):
            curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:])
        else:
            curr_src_latent_codes = get_source_latent_codes_encoder(source_labels[j*interval:(j+1)*interval])
        ret_src_latent_codes.append(curr_src_latent_codes)

    ret_src_latent_codes = torch.cat(ret_src_latent_codes)

# target_data_fol = os.path.join("/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data", "roca_targets_partial", "chair", "h5")
# tid = random.choice(os.listdir(target_data_fol))
# with h5py.File(os.path.join(target_data_fol, tid), 'r') as f:
#     pc = f['points'][:]
# trimesh.PointCloud(pc).export('tst_pc.ply')

"""
Transformation equivalence of annotations and instances
"""
# with open("/mnt/noraid/karacam/Roca/Data/Dataset/Custom5Class/full_annotations_centered.json", 'r') as f:
#     annots = json.load(f)
# with open("/mnt/noraid/karacam/Roca/Data/Dataset/Custom5Class/scan2cad_instances_val.json", 'r') as f:
#     instances = json.load(f)
# _i = np.random.randint(len(instances['annotations']))
# inst = instances['annotations'][_i]
# annot = None
# for _a in annots:
#     if _a['id_scan'] == inst['model']['scene_id']:
#         annot = _a
#         break
# imgs = {img['id']: img['file_name'] for img in instances['images']}
# pose_id = imgs[inst['image_id']].split('/')[-1].split('.')[0]
# with open(f"/mnt/noraid/karacam/Roca/Data/Images/tasks/scannet_frames_25k/{inst['model']['scene_id']}/pose/{pose_id}.txt", 'r') as f:
#     cam2s = f.readlines()
#     cam2s = np.asarray([np.array(t.split(), np.float32) for t in cam2s])
# # with open(f"/mnt/noraid/karacam/Roca/Data/Images/tasks/scannet_frames_25k/{inst['model']['scene_id']}/intrinsics_color.txt", 'r') as f:
# #     K = f.readlines()
# #     K = np.asarray([np.array(t.split(), np.float32) for t in K])

# for m in annot['aligned_models']: 
#     if m['id_cad'] == inst['model']['id_cad']:
#         c2w = make_M_from_tqs(m['trs']['translation'], m['trs']['rotation'], m['trs']['scale'])
#         s2w = make_M_from_tqs(annot['trs']['translation'], annot['trs']['rotation'], annot['trs']['scale'])
#         print(decompose_mat4(np.linalg.inv(cam2s) @ np.linalg.inv(s2w) @ c2w))
# print(decompose_mat4(make_M_from_tqs(inst['t'], inst['q'], inst['s'])))
"""--------------------------------------------------------"""

# with open("/mnt/noraid/karacam/Roca/Data/Dataset/Custom3Class/scan2cad_instances_train.json", 'r') as f:
#     instances_train = json.load(f)
# with open("/mnt/noraid/karacam/Roca/Data/Dataset/Custom3Class/scan2cad_instances_val.json", 'r') as f:
#     instances_val = json.load(f)
# all_ids = set([a['model']['id_cad'] for a in instances_train['annotations']])
# all_ids.update([a['model']['id_cad'] for a in instances_val['annotations']])

# mesh = trimesh.Trimesh(vertices=pd['verts'], faces=pd['faces'])
# mesh.apply_translation(-mesh.bounds.sum(axis=0)/2.)

# mock_ins = {"categories": [{"name": "chair", "id": 0}], "annotations":[a for a in instances['annotations'] if a['model']['catid_cad'] == "03001627"]}
# mock_ins['images'] = []
# obs_imgs = set({})
# imgs = {i['id']: i for i in instances['images']}
# for a in tqdm(mock_ins['annotations']):
#     a["category_id"] = 0
#     if a['image_id'] not in obs_imgs:
#         obs_imgs.add(a['image_id'])
#         mock_ins['images'].append(imgs[a['image_id']])
#     mesh = trimesh.load(os.path.join(shapenet_root_dir, a['model']['catid_cad'], a['model']['id_cad'], "models/model_normalized.obj"), force='mesh')
#     t = a['t']
#     q = a['q']
#     s = a['s']

#     C = mesh.bounds.sum(axis=0)/2.
#     I = np.eye(4)
#     I[:3, 3] += C
#     a['t'], a['q'], a['s'] = decompose_mat4(make_M_from_tqs(t, q, s) @ I)
#     a['t'], a['q'], a['s'] = (a['t'].tolist(), a['q'].tolist(), a['s'].tolist())
# print(len(mock_ins['images']))
# # mock_ins['images'] = [instances['images'][a['image_id']] for a in mock_ins['annotations']]
# with open(f'/mnt/noraid/karacam/Roca/Data/Dataset/CustomCenter/scan2cad_instances_val.json', 'w') as f:
#     json.dump(mock_ins, f)

# instances_file = f'/mnt/noraid/karacam/ThesisData/Dataset/scan2cad_instances_val.json'
# instances_file = f'/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_instances_train.json'
# instances_file = f'/mnt/noraid/karacam/Roca/Data/Dataset/CustomCenter/scan2cad_instances_val.json'
# with open(instances_file, 'rb') as f:
#     instances = json.load(f)
# annots = [a for a in instances['annotations'] if a['model']['catid_cad'] == "03001627"]
# img_ids = [a['image_id'] for a in annots]
# print(len(img_ids))
# print(len(np.unique(img_ids)))
# print(len(instances['images']))
# print(len(np.unique([a['id'] for a in instances['images']])))
# print(len(np.unique([a['file_name'] for a in instances['images']])))
# cid = None
# neq = 0
# il = {i['id']:i for i in instances['images']}
# cat_imgs = [i for i in instances['images'] if i['id'] in img_ids]
# for a in tqdm(annots):
#     if a['image_id'] not in il:
#         neq+=1
# print(neq)
