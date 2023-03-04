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

class JointNet(nn.Module):
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
        self.embedding_dim = 256
        self.TARGET_LATENT_DIM = self.embedding_dim
        self.SOURCE_LATENT_DIM = self.embedding_dim
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
    
    def forward(self, x):
        ret = self.retrieval_encoder(x)
        tar = self.target_encoder(x)
        res = torch.cat([ret, tar], dim=-1)
        return res
    
    # def _embed(self, x):
    #     ret = self.retrieval_encoder(x)
    #     tar = self.target_encoder(x)
    #     return ret, tar
    
    def _embed_trip(
        self,
        pos,
        neg
    ):      
        tar_pos_embeds = self.target_encoder(pos)
        tar_neg_embeds = self.target_encoder(neg)
        
        pos_embeds = self.retrieval_encoder(pos)
        neg_embeds = self.retrieval_encoder(neg)
        return pos_embeds, neg_embeds, tar_pos_embeds, tar_neg_embeds

    
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
    
    def _deform_inference(
        self,
        target_latent_codes,
        reps,
        idx,
        pred_classes,
        scenes: Optional[List[str]] = None,
        has_alignment: Optional[Tensor] = None,
    ): 
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

        # params = torch.stack(all_params)
        # print(idx.shape)
        # print(self.mat[self.source_labels[idx].flatten()].shape)
        return all_params

    def get_source_latent_codes_encoder(self, source_labels):
        source_points = []

        for source_label in source_labels:
            src_points = self.SOURCE_MODEL_INFO[source_label]["points"]	
            source_points.append(src_points)
        source_points = np.array(source_points)
        source_points = torch.from_numpy(source_points).to(self.device, dtype=torch.float)

        src_latent_codes = self.retrieval_encoder(source_points)
        return src_latent_codes

    @property
    def device(self) -> torch.device:
        return 'cuda'