import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import pickle
import numpy as np
import quaternion
import json
import cv2
import trimesh
import h5py
import random
from copy import deepcopy
from tqdm import tqdm
from itertools import product
from trimesh.voxel.creation import local_voxelize
from scipy.spatial import cKDTree as KDTree
import torch
from torch import nn
from pytorch3d.loss import chamfer_distance
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
    TargetEncoder2,
    TargetDecoder,
    ParamDecoder2
)

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
    instances_file = f'/mnt/noraid/karacam/Roca/Data/Dataset/scan2cad_instances_{split}.json'
    with open(instances_file, 'rb') as f:
        instances = json.load(f)

    mock_ins = {"categories": [{"name": "chair", "id": 0}], "annotations":[a for a in instances['annotations'] if a['model']['catid_cad'] == "03001627"][:1000]}
    nid = 0
    obs_ids = {}
    mock_ins['images'] = []
    for a in mock_ins['annotations']:
        a["category_id"] = 0
        file_name = instances['images'][a['image_id']]['file_name']
        if file_name not in obs_ids.keys():
            obs_ids[file_name] = nid
            n_img = {k:v for k, v in instances['images'][a['image_id']].items()}
            n_img['id'] = nid
            mock_ins['images'].append(n_img)
            nid+=1
        a['image_id'] = obs_ids[file_name]
        
    # # mock_ins['images'] = [instances['images'][a['image_id']] for a in mock_ins['annotations']]
    with open(f'/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/scan2cad_instances_{split}_mock.json', 'w') as f:
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
    JOINT_SRC_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data/roca_sources_part_thresh_32_1024p_v1/chair/h5'
    JOINT_MODEL_PATH = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/log/chair522_1024p_v1_icp/model.pth'
    filename_pickle = JOINT_BASE_DIR+"data/generated_datasplits/chair_522_roca_v1.pickle"
    with open(filename_pickle, 'rb') as handle:
        sources = pickle.load(handle)['sources']
    src_data_fol = JOINT_SRC_DIR
    MAX_NUM_PARAMS = -1
    MAX_NUM_PARTS = -1
    ALPHA = 0.1
    USE_SYMMETRY = True
    K = 5
    MARGIN = 10.0
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
    
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/scan2cad_instances_val.json', 'r') as anno_f:
        annos = json.load(anno_f)['annotations']
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/scan2cad_val_scenes.json', 'r') as svf:
        val_scenes = json.load(svf)
    top5_gt = {s:[] for s in val_scenes}
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/points_val.pkl', 'rb') as f:
        points_val = pickle.load(f)
    with open('/mnt/noraid/karacam/Roca/Data/Dataset/Custom25k/val_grids_32.pkl', 'rb') as f:
        grids_val = pickle.load(f)
    assert len({ps['id_cad'] for ps in points_val}) == len(points_val)
    tar_pcs = {(ps['catid_cad'],ps['id_cad']):ps['points'] for ps in points_val}

    ious = []
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

        m_max = np.asarray(_meta['max'])
        m_min = np.asarray(_meta['min'])
        diag = m_max - m_min
        center = (m_max + m_min)/2
        centroid = np.asarray(_meta['centroid'])
        tar_pc = tar_pc + (centroid - center) / np.linalg.norm(diag)
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
        # mesh.vertices = mesh.vertices + (center - centroid) / np.linalg.norm(diag)
        pred_ind = voxelize_mesh(mesh)
        gt_mesh = trimesh.load(f"/mnt/noraid/karacam/ShapeNetCore.v1/03001627/{model_id}/model.obj", force='mesh')
        gt_ind = voxelize_mesh(gt_mesh.apply_transform(trimesh.transformations.euler_matrix(0, np.pi/2,0)))

        # gt_ind = \
        #     grids_val[cat_id, model_id]
        
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
    JOINT_SRC_DIR = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/data/roca_sources_part_thresh_32_1024p_v2/chair/h5'
    JOINT_MODEL_PATH = '/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/log/chair519_1024p_v2/model.pth'
    filename_pickle = JOINT_BASE_DIR+"data/generated_datasplits/chair_519_roca_v2.pickle"
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

# with torch.no_grad():
#     ious = compare_ious()
# proccess_annotations()
# process_400k_custom(split='val')
# joint_input_data(split='val')
# make_mock_instances(split='val')
process_top5_gt()
