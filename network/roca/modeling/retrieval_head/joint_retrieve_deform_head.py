import torch
import torch.nn as nn
import pickle
import sys
import numpy as np
import random
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union
from pytorch3d.loss import chamfer_distance
import time
from collections import defaultdict
from roca.modeling.retrieval_head.joint_retrieve_deform_ops import *
from roca.modeling.retrieval_head.joint_retrieve_deform_modules import (
    TargetEncoder,
    TargetDecoder,
    ParamDecoder2
)
import json

from detectron2.data import (
    MetadataCatalog,
)

Tensor = torch.Tensor
TensorByClass = Dict[int, Tensor]
IDByClass = Dict[int, List[Tuple[str, str]]]
RetrievalResult = Tuple[List[Tuple[str, str]], Tensor]

class JointRetriveDeformHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.has_cads = False
        self.mode = cfg.MODEL.RETRIEVAL_MODE
        self.is_voxel = cfg.INPUT.CAD_TYPE == 'voxel'
        self._loss = cfg.MODEL.RETRIEVAL_LOSS
        self.baseline = False
        self.comp = "comp" in cfg.MODEL.RETRIEVAL_MODE

        self.wild_points_by_class: Optional[TensorByClass] = None
        self.wild_ids_by_class: Optional[IDByClass] = None

        filename_pickle = cfg.JOINT_BASE_DIR+"data/generated_datasplits/chair_522_roca_v1.pickle"
        with open(filename_pickle, 'rb') as handle:
            sources = pickle.load(handle)['sources']
        src_data_fol = cfg.JOINT_SRC_DIR
        self.MAX_NUM_PARAMS = -1
        self.MAX_NUM_PARTS = -1
        self.ALPHA = 0.1
        self.USE_SYMMETRY = True
        self.K = 10
        self.MARGIN = .5
        self.SOURCE_MODEL_INFO = []
        # self.device = 'cuda:0'
        print("Loading joint sources...")
        with open("/home/karacam/Thesis/joint_learning_retrieval_deformation/shape2part_ext.json", 'r') as f:
            shape2part = json.load(f)
        part2shape = dict([(value, key) for key, value in shape2part['chair']['model_names'].items()])

        for i in range(len(sources)):
        # for i in range(50):
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

            curr_num_params = default_param.shape[0]
            if (self.MAX_NUM_PARAMS < curr_num_params):
                self.MAX_NUM_PARAMS = curr_num_params
                self.MAX_NUM_PARTS = int(self.MAX_NUM_PARAMS/6)

            self.SOURCE_MODEL_INFO.append(curr_source_dict)

        print("Done loading joint sources.")
        self.embedding_size = 6
        self.TARGET_LATENT_DIM = 256
        self.SOURCE_LATENT_DIM = 256
        self.PART_LATENT_DIM = 32
        # with torch.no_grad():
        self.target_encoder = TargetEncoder(
            self.TARGET_LATENT_DIM,
            3,
        )
        decoder_input_dim = self.TARGET_LATENT_DIM + self.SOURCE_LATENT_DIM + self.PART_LATENT_DIM
        self.param_decoder = ParamDecoder2(decoder_input_dim, 256, self.embedding_size)
        # param_decoder.to(device, dtype=torch.float)
        self.retrieval_encoder = TargetEncoder(
            self.TARGET_LATENT_DIM,
            3,
        )
        if self._loss == 'triplet':
            self.embed_loss = nn.TripletMarginLoss(margin=self.MARGIN)
        else:
            self.bce_loss = nn.BCELoss()
        if self.comp:
            self.comp_loss = nn.MSELoss()
            self.comp_encoder = TargetEncoder(
                self.TARGET_LATENT_DIM,
                4,
            )
            self.comp_decoder = TargetDecoder(self.TARGET_LATENT_DIM)

        _model = torch.load(cfg.JOINT_MODEL_PATH)
        self.target_encoder.load_state_dict(_model["target_encoder"])
        self.target_encoder.to(self.device)
        self.target_encoder.eval()

        self.param_decoder.load_state_dict(_model["param_decoder"])
        self.param_decoder.to(self.device)
        self.param_decoder.eval()

        self.retrieval_encoder.load_state_dict(_model["retrieval_encoder"])
        self.retrieval_encoder.to(self.device)
        self.retrieval_encoder.eval()

        self.SOURCE_LATENT_CODES = _model["source_latent_codes"].detach()
        self.SOURCE_PART_LATENT_CODES = [_x.detach() for _x in _model["part_latent_codes"]]

        self.SOURCE_VARIANCES = _model["source_variances"].detach()

        self.source_labels = torch.arange(len(self.SOURCE_MODEL_INFO))#.repeat(10)
        src_mats, src_default_params, src_connectivity_mat = get_source_info(self.source_labels, self.SOURCE_MODEL_INFO, self.MAX_NUM_PARAMS, use_connectivity=True)
        self.src_latent_codes = torch.gather(self.SOURCE_LATENT_CODES, 0, self.source_labels.to(self.device).unsqueeze(-1).repeat(1,self.SOURCE_LATENT_CODES.shape[-1]))

        self.mat = torch.stack([mat for mat in src_mats])#.to(device, dtype=torch.float)
        self.def_param = torch.stack([def_param for def_param in src_default_params])#.to(device, dtype=torch.float)
        self.conn_mat = torch.stack([conn_mat for conn_mat in src_connectivity_mat])

        with torch.no_grad():
            self.ret_src_latent_codes = []
            num_sets = 20
            interval = int(len(self.source_labels)/num_sets)

            for j in range(num_sets):
                if (j==num_sets-1):
                    curr_src_latent_codes = self.get_source_latent_codes_encoder(self.source_labels[j*interval:])
                else:
                    curr_src_latent_codes = self.get_source_latent_codes_encoder(self.source_labels[j*interval:(j+1)*interval])
                self.ret_src_latent_codes.append(curr_src_latent_codes)

            self.ret_src_latent_codes = torch.cat(self.ret_src_latent_codes)#.view(len(self.SOURCE_MODEL_INFO), -1, self.SOURCE_LATENT_DIM)
    
    def forward(
        self,
        noc_points: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        pos_cads: Optional[Tensor] = None,
        neg_cads: Optional[Tensor] = None,
        classes: Optional[Tensor] = None,
        scenes: Optional[List[str]] = None,
        instance_sizes: Optional[List[int]] = None,
        has_alignment: Optional[Tensor] = None,
        shape_code: Optional[Tensor] = None,
    ):

        # x = nn.functional.interpolate(noc_points, scale_factor=3, mode='bilinear', align_corners=True)
        # _masks = nn.functional.interpolate(masks, scale_factor=3, mode='bilinear', align_corners=True).gt(0.3)

        # nocs_sampled = []
        # pos_cads_sampled = []
        # neg_cads_sampled = []
        # if self.training:
        #     for ns, pc, nc, ms in zip(noc_points, pos_cads, neg_cads, masks):
        #         _x = ns[:, ms.type(torch.bool).squeeze(0)]
        #         if _x.shape[1] == 0:
        #             print("Encountered 0 sampled noc!!!")
        #             nocs_sampled.append(ns.view(3,-1)[:, torch.randperm(ns.shape[1]*ns.shape[2])[:1024]])
        #         elif 1024 <= _x.shape[1]:
        #             nocs_sampled.append(_x[:, torch.randperm(_x.shape[1])[:1024]])
        #         else:
        #             nocs_sampled.append(_x.repeat(1,(1024//_x.shape[0])+1)[:,:1024])
        #         pos_cads_sampled.append(pc)
        #         neg_cads_sampled.append(nc)
        #         # print(torch.gather(ns, 0, torch.stack(torch.where(ms))).shape)
        #     if not nocs_sampled:
        #         return torch.tensor(0.0), torch.tensor(0.0)
        #     nocs_sampled = torch.stack(nocs_sampled)
        #     pos_cads_sampled = torch.stack(pos_cads_sampled).transpose(1,2)
        #     neg_cads_sampled = torch.stack(neg_cads_sampled).transpose(1,2)
        # else:
        #     for ns, ms in zip(x, _masks):
        #         _x = ns[:, ms.squeeze(0)]
        #         if _x.shape[1] == 0:
        #             print("Encountered 0 sampled noc!!!")
        #             nocs_sampled.append(torch.zeros((3,1024), dtype=torch.float, device=self.device))
        #         elif 1024 <= _x.shape[1]:
        #             nocs_sampled.append(_x[:, torch.randperm(_x.shape[1])[:1024]])
        #         else:
        #             nocs_sampled.append(_x.repeat(1,(1024//_x.shape[0])+1)[:,:1024])
        #     nocs_sampled = torch.stack(nocs_sampled).transpose(1,2)
        # x = torch.gather(x, 1, torch.nonzero((x!=0).any(1)).view(x.shape[0],-1,x.shape[1]))
        # print(x.shape)

        _reps = noc_points.shape[0]
        nocs_sampled = noc_points.view(_reps, 3, -1)

        if self.comp:
            comp_latent_codes = self.comp_encoder(torch.cat([masks.view(_reps,1,-1), nocs_sampled],dim=1).transpose(1,2))
            nocs_comp = self.comp_decoder(comp_latent_codes)
        else:
            nocs_comp = nocs_sampled.transpose(1,2)
        target_latent_codes = self.target_encoder(nocs_comp)
        retrieval_latent_codes = self.retrieval_encoder(nocs_comp)

        # retrieval_latent_codes = retrieval_latent_codes.unsqueeze(0).repeat(len(self.SOURCE_MODEL_INFO),1,1)
        # retrieval_latent_codes = retrieval_latent_codes.view(-1, retrieval_latent_codes.shape[-1])
        # x_repeated = nocs_sampled.unsqueeze(0).repeat(len(self.SOURCE_MODEL_INFO),1,1,1)
        # src_latent_codes = self.src_latent_codes.repeat(_reps,1)
        # source_labels = self.source_labels.repeat(_reps)
        if self.training:
            sample = torch.randperm(_reps)[:32]
            nocs_comp = nocs_sampled[sample]
            pos_cads_sampled = pos_cads[sample]
            neg_cads_sampled = neg_cads[sample]
            
            target_latent_codes = target_latent_codes[sample]
            tar_pos_embeds = self.target_encoder(pos_cads_sampled)
            tar_neg_embeds = self.target_encoder(neg_cads_sampled)
            
            retrieval_latent_codes = retrieval_latent_codes[sample]
            pos_embeds = self.retrieval_encoder(pos_cads_sampled)
            neg_embeds = self.retrieval_encoder(neg_cads_sampled)
            comp_loss = None
            if self.comp:
                comp_loss, _ = chamfer_distance(nocs_comp, pos_cads_sampled)
                # comp_loss = self.comp_loss(nocs_comp, pos_cads_sampled)
            if self._loss == 'triplet':
                embedding_loss = self.embed_loss(retrieval_latent_codes, pos_embeds, neg_embeds)+self.embed_loss(target_latent_codes,tar_pos_embeds,tar_neg_embeds)
                # return embedding_loss/2.
                return comp_loss, embedding_loss/2.
            else:
                # return self._retrieval(retrieval_latent_codes, pos_embeds, min(32,_reps))
                return comp_loss, self._retrieval(retrieval_latent_codes, pos_embeds, min(32,_reps))
        else:
            # target_latent_codes = self.target_encoder(nocs_sampled)
            scenes = list(chain(*(
                [scene] * isize
                for scene, isize in zip(scenes, instance_sizes)
            )))
            cad_ids = [None for _ in scenes]
            params = [None for _ in scenes]
            retrieved_idx = [None for _ in scenes]
            for scene in set(scenes):
                scene_mask = [scene_ == scene for scene_ in scenes]
                scene_noc_embeds = retrieval_latent_codes[scene_mask]
                scene_tar_embeds = target_latent_codes[scene_mask]
                scene_classes = classes[scene_mask]

                retrieved_idx_scene, cad_ids_scene = self._retrieval_inference(scene_noc_embeds, scene_noc_embeds.shape[0])
                params_scene = self._deform_inference(scene_tar_embeds, scene_noc_embeds.shape[0], retrieved_idx_scene)
                cad_ids_scene.reverse()
                params_scene.reverse()
                retrieved_idx_scene.reverse()
                for i, m in enumerate(scene_mask):
                    if m:
                        cad_ids[i] = cad_ids_scene.pop()
                        params[i] = params_scene.pop()
                        retrieved_idx[i] = retrieved_idx_scene.pop()
            has_alignment[[id is None for id in cad_ids]] = False
            # retrieved_idx, cad_ids = self._retrieval_inference(retrieval_latent_codes, _reps)
            # params = self._deform_inference(target_latent_codes, _reps, retrieved_idx, scenes=scenes, has_alignment=has_alignment, pred_classes=classes)
            pred_indices = torch.arange(classes.numel(), dtype=torch.long)
            return cad_ids, pred_indices, params, retrieved_idx, torch.cat([nocs_comp.unsqueeze(1), nocs_sampled.transpose(1,2).unsqueeze(1)], dim=1)

    def _retrieval(
        self,
        noc_embeds,
        pos_embeds,
        reps
    ):
        retrieval_latent_codes = noc_embeds.unsqueeze(0).repeat(len(self.SOURCE_MODEL_INFO),1,1)
        retrieval_latent_codes = retrieval_latent_codes.view(len(self.SOURCE_MODEL_INFO), -1, self.TARGET_LATENT_DIM)
        pos_latent_codes = pos_embeds.unsqueeze(0).repeat(len(self.SOURCE_MODEL_INFO),1,1)
        pos_latent_codes = pos_latent_codes.view(len(self.SOURCE_MODEL_INFO), -1, self.TARGET_LATENT_DIM)
        with torch.no_grad():
            source_labels = self.source_labels.repeat(reps)
            _src_latent_codes = self.ret_src_latent_codes.repeat(reps,1).view(len(self.SOURCE_MODEL_INFO), -1, self.SOURCE_LATENT_DIM)

            src_variances = get_source_latent_codes_fixed(source_labels, self.SOURCE_VARIANCES, device=self.device)
            src_variances = src_variances.view(len(self.SOURCE_MODEL_INFO), -1, self.SOURCE_LATENT_DIM)

        distances = compute_mahalanobis(retrieval_latent_codes, _src_latent_codes, src_variances, activation_fn=torch.sigmoid)
        gt_distances = compute_mahalanobis(pos_latent_codes, _src_latent_codes, src_variances, activation_fn=torch.sigmoid)
        sorted_indices = torch.argsort(distances, dim=0)
        retrieved_idx = sorted_indices[0,:]
        gt_sorted_indices = torch.argsort(gt_distances, dim=0)
        gt_retrieved_idx = gt_sorted_indices[:self.K,:]
        return self.bce_loss((retrieved_idx == gt_retrieved_idx).any(dim=0).type(torch.float), torch.ones_like(retrieved_idx, dtype=torch.float))/100.
    
    def _retrieval_inference(
        self,
        retrieval_latent_codes,
        reps
    ):
        retrieval_latent_codes = retrieval_latent_codes.unsqueeze(0).repeat(len(self.SOURCE_MODEL_INFO),1,1)
        # retrieval_latent_codes = retrieval_latent_codes.view(-1, retrieval_latent_codes.shape[-1])	
        retrieval_latent_codes = retrieval_latent_codes.view(len(self.SOURCE_MODEL_INFO), -1, self.TARGET_LATENT_DIM)
        with torch.no_grad():
            source_labels = self.source_labels.repeat(reps)
            _src_latent_codes = self.ret_src_latent_codes.repeat(reps,1).view(len(self.SOURCE_MODEL_INFO), -1, self.SOURCE_LATENT_DIM)

            src_variances = get_source_latent_codes_fixed(source_labels, self.SOURCE_VARIANCES, device=self.device)
            src_variances = src_variances.view(len(self.SOURCE_MODEL_INFO), -1, self.SOURCE_LATENT_DIM)

        distances = compute_mahalanobis(retrieval_latent_codes, _src_latent_codes, src_variances, activation_fn=torch.sigmoid)
        sorted_indices = torch.argsort(distances, dim=0)
        retrieved_idx = sorted_indices[0,:]

        cad_ids = [self.SOURCE_MODEL_INFO[sl]['model_id'] for sl in retrieved_idx]
        return retrieved_idx.tolist(), cad_ids

    def _deform(
        self,
        target_latent_codes,
        idx,
        nocs_sampled,
    ):  
        x_repeated = nocs_sampled.unsqueeze(0).repeat(self.K,1,1,1)
        x_repeated = x_repeated.view(-1, x_repeated.shape[-2], x_repeated.shape[-1])

        src_latent_codes = self.src_latent_codes[self.source_labels[idx].flatten()]
        concat_latent_code = torch.cat((src_latent_codes, target_latent_codes.unsqueeze(0).repeat(self.K,1,1).view(-1, target_latent_codes.shape[-1])), dim=1)

        all_params = []
        # source_labels = self.source_labels.repeat(reps)
        for j in range(concat_latent_code.shape[0]):
            curr_num_parts = self.SOURCE_MODEL_INFO[self.source_labels[j]]["num_parts"]
            curr_code = concat_latent_code[j]
            curr_code_repeated = curr_code.view(1,curr_code.shape[0]).repeat(curr_num_parts, 1)
            
            part_latent_codes = self.SOURCE_PART_LATENT_CODES[self.source_labels[j]]

            full_latent_code = torch.cat((curr_code_repeated, part_latent_codes), dim=1)

            params = self.param_decoder(full_latent_code, use_bn=False)

            ## Pad with extra zero rows to cater to max number of parameters
            if (curr_num_parts < self.MAX_NUM_PARTS):
                dummy_params = torch.zeros((self.MAX_NUM_PARTS-curr_num_parts, self.embedding_size), dtype=torch.float, device=self.device)
                params = torch.cat((params, dummy_params), dim=0)

            params = params.view(-1, 1)
            all_params.append(params)

        params = torch.stack(all_params)
        # print(idx.shape)
        # print(self.mat[self.source_labels[idx].flatten()].shape)
        output_pcs = get_shape(
            self.mat[self.source_labels[idx].flatten()].to(self.device), 
            params, 
            self.def_param[self.source_labels[idx].flatten()].to(self.device), 
            self.ALPHA, 
            connectivity_mat=self.conn_mat[self.source_labels[idx].flatten()].to(self.device)
        )
        # print(output_pcs.shape)
        # print(x_repeated.shape)
        cd_loss, _ = chamfer_distance(x_repeated.transpose(1,2), output_pcs, batch_reduction=None)

        ## Get the min loss from the different sources
        loss = cd_loss.view(self.K, -1)
        fitting_loss = torch.mean(loss)
        if self.USE_SYMMETRY:
            reflected_pc = get_symmetric(output_pcs)
            symmetric_loss, _ = chamfer_distance(output_pcs, reflected_pc)
            fitting_loss += symmetric_loss
        return fitting_loss, output_pcs
    
    def _deform_inference(
        self,
        target_latent_codes,
        reps,
        idx,
    ):  
        # print(target_latent_codes.shape)
        # print(self.points_by_class[0].shape)
        # self.wild_ids_by_class
        # print(self.cad_ids_by_class)
        # print(self.indices_by_scene)
        src_latent_codes = self.src_latent_codes[idx]
        concat_latent_code = torch.cat((src_latent_codes, target_latent_codes.view(-1, target_latent_codes.shape[-1])), dim=1)

        # src_latent_codes = self.src_latent_codes.repeat(reps,1)
        # concat_latent_code = torch.cat((src_latent_codes, target_latent_codes.repeat(len(self.SOURCE_MODEL_INFO),1,1).view(-1, target_latent_codes.shape[-1])), dim=1)

        all_params = []
        # source_labels = self.source_labels.repeat(reps)
        for j in range(concat_latent_code.shape[0]):
            curr_num_parts = self.SOURCE_MODEL_INFO[self.source_labels[j]]["num_parts"]
            curr_code = concat_latent_code[j]
            curr_code_repeated = curr_code.view(1,curr_code.shape[0]).repeat(curr_num_parts, 1)
            
            part_latent_codes = self.SOURCE_PART_LATENT_CODES[self.source_labels[j]]

            full_latent_code = torch.cat((curr_code_repeated, part_latent_codes), dim=1)

            params = self.param_decoder(full_latent_code, use_bn=False)

            ## Pad with extra zero rows to cater to max number of parameters
            if (curr_num_parts < self.MAX_NUM_PARTS):
                dummy_params = torch.zeros((self.MAX_NUM_PARTS-curr_num_parts, self.embedding_size), dtype=torch.float, device=self.device)
                params = torch.cat((params, dummy_params), dim=0)

            params = params.view(-1, 1)
            all_params.append(params)

        params = torch.stack(all_params)
        # print(idx.shape)
        # print(self.mat[self.source_labels[idx].flatten()].shape)
        # cd_loss, _ = chamfer_distance(output_pcs, x_repeated.to(device, dtype=torch.float), batch_reduction=None)
        # cd_loss = cd_loss.view(len(SOURCE_MODEL_INFO), -1)#.detach().item()

        return all_params

    def get_source_latent_codes_encoder(self, source_labels):
        # print("Using encoder to get source latent codes.")
        source_points = []

        # start_tm = time.time()
        for source_label in source_labels:
            src_points = self.SOURCE_MODEL_INFO[source_label]["points"]	
            source_points.append(src_points)
        # ret_tm = time.time()
        # print("Load from labels time: ", ret_tm - start_tm)
        # print("Num labels: ", source_labels.shape)
        source_points = np.array(source_points)
        source_points = torch.from_numpy(source_points).to(self.device, dtype=torch.float)

        src_latent_codes = self.retrieval_encoder(source_points)
        # print("Retrieval encoding time: ", time.time() - ret_tm)
        return src_latent_codes

    def inject_cad_models(
        self,
        points: TensorByClass,
        ids: IDByClass,
        scene_data: Dict[str, List[Dict[str, Any]]],
        device: Union[torch.device, str] = 'cpu'
    ):
        # self.device = device
        self.has_cads = True
        self.points_by_class = {k: torch.Tensor(v).to(self.device) for k, v in points.items()}
        self.cad_ids_by_class = ids
        # self.dummy_mesh = ico_sphere()

        # Parse scene data
        classes = list(self.cad_ids_by_class.keys())
        scene_by_ids = defaultdict(lambda: [])
        for scene, models in scene_data.items():
            for model in models:
                model_id = (model['catid_cad'], model['id_cad'])
                scene_by_ids[model_id].append(scene)

        self.indices_by_scene = {
            scene: {k: [] for k in classes}
            for scene in scene_data.keys()
        }
        for k in classes:
            for i, cad_id in enumerate(self.cad_ids_by_class[k]):
                scenes = scene_by_ids[cad_id]
                for scene in scenes:
                    self.indices_by_scene[scene][k].append(i)

    def eject_cad_models(self):
        if self.has_cads:
            del self.points_by_class
            del self.cad_ids_by_class
            del self.indices_by_scene
            self.has_cads = False
    
    def embed_cads(self, cad_points: Tensor) -> Tensor:
        if self.baseline:
            return cad_points
        elif self.is_voxel:
            return self.target_encoder(cad_points.float())
        else:  # Point clouds
            return cad_points.transpose(-2, -1)

    @property
    def device(self) -> torch.device:
        return 'cuda'
    
    @property
    def has_wild_cads(self) -> bool:
        return self.wild_points_by_class is not None