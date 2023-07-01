from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from roca.modeling.retrieval_head.retrieval_modules import (
    PointNet,
    ResNetDecoder,
    ResNetEncoder,
)
from roca.modeling.retrieval_head.joint_e2e_modules import (
    JointNet
)
from roca.modeling.retrieval_head.retrieval_ops import (
    embedding_lookup,
    grid_to_point_list,
    nearest_points_retrieval,
    random_retrieval,
    voxelize_nocs,
)
from roca.modeling.retrieval_head.joint_retrieve_deform_ops import (
    get_symmetric,
    regression_loss
)

from pytorch3d.loss import chamfer_distance


Tensor = torch.Tensor
TensorByClass = Dict[int, Tensor]
IDByClass = Dict[int, List[Tuple[str, str]]]
RetrievalResult = Tuple[List[Tuple[str, str]], Tensor]


class JointE2EHead(nn.Module):
    def __init__(self, cfg, shape_code_size: int, margin: float = .5):
        super().__init__()
        self.has_cads = False
        self.mode = cfg.MODEL.RETRIEVAL_MODE
        self.loss_type = cfg.MODEL.RETRIEVAL_LOSS
        self.shape_code_size = shape_code_size
        self.is_voxel = cfg.INPUT.CAD_TYPE == 'voxel'

        # NOTE: Make them embeddings for the learned model
        self.wild_points_by_class: Optional[TensorByClass] = None
        self.wild_ids_by_class: Optional[IDByClass] = None

        self.num_classes = 5

        self.baseline = cfg.MODEL.RETRIEVAL_BASELINE
        if self.baseline:
            return

        if self.loss_type == 'regression':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'triplet':
            self.loss = nn.TripletMarginLoss(margin=margin)
        else:
            self.loss_reg = nn.MSELoss()
            self.loss_trip = nn.TripletMarginLoss(margin=margin)
        self._sigmoid = nn.Sigmoid()

        if '_' in self.mode:
            self.cad_mode, self.noc_mode = self.mode.split('_')
        else:
            self.cad_mode = self.noc_mode = self.mode

        if self.cad_mode == 'pointnet':
            assert not self.is_voxel, 'Inconsistent CAD modality'
            self.cad_net = PointNet()
        elif self.cad_mode == 'joint':
            self.joint_net = {}
            for i, _num_s in zip([0,1,2,3,4], [522,676,306,138,79]):
                self.joint_net[i] = JointNet(cfg, i, _num_s)
            self.cad_net_ret = PointNet()
            self.cad_net_tar = PointNet()
            # self.cad_net = JointNet(cfg)
        elif self.cad_mode == 'resnet':
            assert self.is_voxel, 'Inconsistent CAD modality'
            self.cad_net = ResNetEncoder()
        else:
            raise ValueError(
                'Unknown CAD network type {}'.format(self.cad_mode)
            )

        if self.noc_mode == 'pointnet':
            self.noc_net = PointNet()
        elif self.noc_mode in ('joint', "joint+e2e"):
            self.noc_net_ret = nn.ModuleDict({
                'pointnet': PointNet(fc_out=True, joint=True, num_classes=self.num_classes),
                'image': self.make_image_mlp(embed_dim=256*self.num_classes)
            })
            self.noc_net_tar = nn.ModuleDict({
                'pointnet': PointNet(fc_out=True, joint=True, num_classes=self.num_classes),
                'image': self.make_image_mlp(embed_dim=256*self.num_classes)
            })
        elif self.noc_mode == 'joint+sep':
            self.noc_net_ret = nn.ModuleDict()
            self.noc_net_tar = nn.ModuleDict()
            for i in range(self.num_classes):
                self.noc_net_ret.update({
                    f'pointnet_{i}': PointNet(fc_out=True),
                    f'image_{i}': self.make_image_mlp(embed_dim=256)
                })
                self.noc_net_tar.update({
                    f'pointnet_{i}': PointNet(fc_out=True),
                    f'image_{i}': self.make_image_mlp(embed_dim=256)
                })
        elif self.noc_mode == 'image':
            self.noc_net = self.make_image_mlp()
        elif self.noc_mode == 'pointnet+image':
            self.noc_net = nn.ModuleDict({
                'pointnet': PointNet(),
                'image': self.make_image_mlp()
            })
        elif self.noc_mode == 'resnet':
            self.noc_net = ResNetEncoder()
        elif self.noc_mode == 'resnet+image':
            self.noc_net = nn.ModuleDict({
                'resnet': ResNetEncoder(),
                'image': self.make_image_mlp(embed_dim=256)
            })
        elif self.noc_mode in ('resnet+image+comp', 'resnet+image+fullcomp'):
            resnet = ResNetEncoder()
            self.noc_net = nn.ModuleDict({
                'resnet': resnet,
                'image': self.make_image_mlp(embed_dim=256),
                'comp': ResNetDecoder(relu_in=True, feats=resnet.feats)
            })
            self.comp_loss = nn.BCELoss()
        else:
            raise ValueError('Unknown noc mode {}'.format(self.noc_mode))

    def make_image_mlp(self, relu_out: bool = True, embed_dim=1024) -> nn.Module:
        # if self.noc_mode in ('joint', "joint+e2e"):
        #     embed_dim = self.joint_net.embedding_dim
        # elif self.cad_mode == 'joint':
        #     embed_dim = self.cad_net_ret.embedding_dim
        # else:
        #     embed_dim = self.cad_net.embedding_dim
        return nn.Sequential(
            nn.Linear(self.shape_code_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, embed_dim),
            nn.ReLU(True) if relu_out else nn.Identity()
        )

    @property
    def has_wild_cads(self) -> bool:
        return self.wild_points_by_class is not None

    def inject_cad_models(
        self,
        points: TensorByClass,
        ids: IDByClass,
        scene_data: Dict[str, List[Dict[str, Any]]],
        device: Union[torch.device, str] = 'cpu'
    ):
        self.device = device
        self.has_cads = True
        if self.is_voxel:
            self.points_by_class = points
        else:
            self.points_by_class = {k: v.to(device) for k, v in points.items()}
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

    def forward(
        self,
        classes: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        noc_points: Optional[Tensor] = None,
        shape_code: Optional[Tensor]= None,
        instance_sizes: Optional[List[int]] = None,
        has_alignment: Optional[Tensor] = None,
        scenes: Optional[List[str]] = None,
        wild_retrieval: bool = False,
        pos_cads: Optional[Tensor] = None,
        neg_cads: Optional[Tensor] = None
    ) -> Union[Dict[str, Tensor], RetrievalResult]:

        if self.training:
            losses = {}
            if self.baseline:
                return losses

            assert pos_cads is not None
            assert neg_cads is not None

            if self.cad_mode == 'joint':
                losses['loss_embedding_ret'] = 0
                losses['loss_embedding_tar'] = 0
                if self.noc_mode == 'joint+sep':
                    for i in range(self.num_classes):
                        class_mask = classes == i
                        if not class_mask.any():
                            continue
                        ret_noc_embed = self.noc_net_ret[f'pointnet_{i}'](noc_points[class_mask], masks[class_mask]) + self.noc_net_ret[f'image_{i}'](shape_code[class_mask])
                        tar_noc_embed = self.noc_net_tar[f'pointnet_{i}'](noc_points[class_mask].detach(), masks[class_mask].detach()) + self.noc_net_tar[f'image_{i}'](shape_code[class_mask].detach())
                        ret_pos_embed, tar_pos_embed = self.joint_net[i]._embed(pos_cads[class_mask])
                        losses['loss_embedding_ret'] += self.loss(
                            ret_noc_embed,
                            ret_pos_embed
                        ) / class_mask.sum()
                        losses['loss_embedding_tar'] += self.loss(
                            tar_noc_embed, 
                            tar_pos_embed
                        )
                else:
                    ret_noc_embed = self.noc_net_ret['pointnet'](noc_points, masks) + self.noc_net_ret['image'](shape_code)
                    tar_noc_embed = self.noc_net_tar['pointnet'](noc_points.detach(), masks.detach()) + self.noc_net_tar['image'](shape_code.detach())
                    for i in range(self.num_classes):
                        class_mask = classes == i
                        if not class_mask.any():
                            continue
                        ret_pos_embed, tar_pos_embed = self.joint_net[i]._embed(pos_cads)
                        losses['loss_embedding_ret'] += self.loss(ret_noc_embed[class_mask][...,i*256:(i+1)*256], ret_pos_embed[class_mask]) #/ class_mask.sum()
                        losses['loss_embedding_tar'] += self.loss(tar_noc_embed[class_mask][...,i*256:(i+1)*256], tar_pos_embed[class_mask])

            else:
                noc_embed = self.embed_nocs(
                    shape_code=shape_code,
                    noc_points=noc_points,
                    mask=masks
                )
                if isinstance(noc_embed, tuple):  # Completion
                    noc_embed, noc_comp = noc_embed
                    losses['loss_noc_comp'] = self.comp_loss(
                        noc_comp, pos_cads.to(dtype=noc_comp.dtype)
                    )
                cad_embeds = self.embed_cads(torch.cat([pos_cads, neg_cads]))
                pos_embed, neg_embed = torch.chunk(cad_embeds, 2)
                losses['loss_triplet'] = self.loss(noc_embed, pos_embed, neg_embed)
            return losses

        else:  # Lookup for CAD ids at inference
            if wild_retrieval:
                assert self.has_wild_cads, 'No registered wild CAD models'
            else:
                assert self.has_cads, 'No registered CAD models!'

            scenes = list(chain(*(
                [scene] * isize
                for scene, isize in zip(scenes, instance_sizes)
            )))

            if self.baseline:
                return self._perform_baseline(
                    has_alignment,
                    classes,
                    masks,
                    scenes,
                    noc_points,
                    wild_retrieval=wild_retrieval
                )
            else:
                return self._embedding_lookup(
                    has_alignment,
                    classes,
                    masks,
                    scenes,
                    noc_points,
                    wild_retrieval=wild_retrieval,
                    shape_code=shape_code
                )

    def embed_nocs(
        self,
        shape_code: Optional[Tensor] = None,
        noc_points: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        # Assertions
        if 'image' in self.noc_mode:
            assert shape_code is not None
        if self.noc_mode != 'image':
            assert noc_points is not None
            assert mask is not None

        if self.is_voxel:
            noc_points = voxelize_nocs(grid_to_point_list(noc_points, mask))

        if self.noc_mode == 'image':
            return self.noc_net(shape_code)
        elif self.noc_mode == 'pointnet':
            return self.noc_net(noc_points, mask)
        elif self.noc_mode == 'pointnet+image':
            return (
                self.noc_net['pointnet'](noc_points, mask)
                + self.noc_net['image'](shape_code)
            )
        elif self.noc_mode == 'resnet':
            return self.noc_net(noc_points)
        elif self.noc_mode == 'resnet+image':
            return (
                self.noc_net['resnet'](noc_points)
                + self.noc_net['image'](shape_code)
            )
        elif self.noc_mode in ('resnet+image+comp', 'resnet+image+fullcomp'):
            noc_embed = self.noc_net['resnet'](noc_points)
            result = noc_embed + self.noc_net['image'](shape_code)
            if self.training:
                if self.noc_mode == 'resnet+image+comp':
                    comp = self.noc_net['comp'](noc_embed)
                else:  # full comp
                    comp = self.noc_net['comp'](result)
                return result, comp.sigmoid_()
            else:
                return result
        elif self.noc_mode in ('joint+resnet+image+comp', 'joint+resnet+image+fullcomp'):
            noc_embed_ret = self.noc_net_ret['resnet'](noc_points)
            result_ret = noc_embed_ret + self.noc_net_ret['image'](shape_code)
            if self.training:
                if self.noc_mode == 'resnet+image+comp':
                    comp_ret = self.noc_net_ret['comp'](noc_embed_ret)
                else:  # full comp
                    comp_ret = self.noc_net_ret['comp'](result_ret)

            noc_embed_tar = self.noc_net_tar['resnet'](noc_points)
            result_tar = noc_embed_tar + self.noc_net_tar['image'](shape_code)
            if self.training:
                if self.noc_mode == 'resnet+image+comp':
                    comp_tar = self.noc_net_tar['comp'](noc_embed_tar)
                else:  # full comp
                    comp_tar = self.noc_net_tar['comp'](result_tar)
                return result_ret, comp_ret.sigmoid_(), result_tar, comp_tar.sigmoid_()
            else:
                return result_ret, result_tar
        else:
            raise ValueError('Unknown noc embedding type {}'
                             .format(self.noc_mode))

    def embed_cads(self, cad_points: Tensor) -> Tensor:
        if self.baseline:
            return cad_points
        elif self.is_voxel:
            return self.cad_net(cad_points.float())
        elif self.cad_mode == 'joint':
            return self.joint_net[0](cad_points)
        else:  # Point clouds
            return self.cad_net(cad_points.transpose(-2, -1))

    def _perform_baseline(
        self,
        has_alignment,
        pred_classes,
        pred_masks,
        scenes,
        noc_points=None,
        wild_retrieval=False
    ):
        num_instances = pred_classes.numel()
        if has_alignment is None:
            has_alignment = torch.ones(num_instances, dtype=torch.bool)

        if self.mode == 'nearest':
            function = nearest_points_retrieval
        elif self.mode == 'random':
            function = random_retrieval
        elif self.mode == 'first':
            function = 'first'
        else:
            raise ValueError('Unknown retrieval mode: {}'.format(self.mode))

        # meshes = []
        ids = []
        j = -1
        for i, scene in enumerate(scenes):
            if not has_alignment[i].item():
                # meshes.append(self.dummy_mesh)
                ids.append(None)
                continue
            j += 1

            pred_class = pred_classes[j].item()

            if wild_retrieval:
                assert self.wild_points_by_class is not None
                points_by_class = self.wild_points_by_class[pred_class]
                cad_ids_by_class = self.wild_ids_by_class[pred_class]
            else:
                points_by_class = self.points_by_class[pred_class]
                point_indices = self.indices_by_scene[scene][pred_class]
                if len(point_indices) == 0:
                    # meshes.append(self.dummy_mesh)
                    ids.append(None)
                    has_alignment[i] = False  # No CAD -> No Alignment
                    continue
                points_by_class = points_by_class[point_indices]
                cad_ids_by_class = self.cad_ids_by_class[pred_class]

            if function is nearest_points_retrieval:
                assert noc_points is not None
                index, _ = function(
                    noc_points[j],
                    pred_masks[j],
                    points_by_class,
                    use_median=False,  # True
                    # mask_probs=mask_probs[j]
                )
            elif isinstance(function, str) and function == 'first':
                index = torch.zeros(1).int()
            elif function is random_retrieval:
                index, _ = random_retrieval(points_by_class)
            else:
                raise ValueError('Unknown baseline {}'.format(function))

            index = index.item()
            if not wild_retrieval:
                index = point_indices[index]
            ids.append(cad_ids_by_class[index])

        # Model ids
        cad_ids = ids
        # To handle sorting and filtering of instances
        pred_indices = torch.arange(num_instances, dtype=torch.long)
        return cad_ids, pred_indices

    def _embedding_lookup(
        self,
        has_alignment,
        pred_classes,
        pred_masks,
        scenes,
        noc_points,
        wild_retrieval,
        shape_code
    ):
        if wild_retrieval:
            noc_embeds = self.embed_nocs(shape_code, noc_points, pred_masks)
            cad_ids = embedding_lookup(
                pred_classes,
                noc_embeds,
                self.wild_points_by_class,
                self.wild_ids_by_class
            )

        else:
            assert scenes is not None
            assert has_alignment is not None

            cad_ids = [None for _ in scenes]
            params = [None for _ in scenes]
            retrieved_idx = [None for _ in scenes]
            if 'joint' == self.cad_mode:
                _reps = noc_points.shape[0]
                
                nocs_sampled = noc_points.view(_reps, 3, -1)
                if not self.noc_mode == 'joint+sep':
                    ret_noc_embed = self.noc_net_ret['pointnet'](noc_points, pred_masks) + self.noc_net_ret['image'](shape_code)
                    tar_noc_embed = self.noc_net_tar['pointnet'](noc_points, pred_masks) + self.noc_net_tar['image'](shape_code)
                for scene in set(scenes):
                    scene_mask = [scene_ == scene for scene_ in scenes]
                    scene_classes = pred_classes[scene_mask]

                    cad_ids_scene = [None for _ in scene_classes]
                    params_scene = [None for _ in scene_classes]
                    retrieved_idx_scene = [None for _ in scene_classes]
                    if self.noc_mode == 'joint+sep':
                        for c_cat in range(self.num_classes):
                            cat_mask = scene_classes == c_cat
                            if cat_mask.sum() == 0:
                                continue
                            scene_ret_noc_embed = self.noc_net_ret[f'pointnet_{c_cat}'](noc_points[cat_mask], pred_masks[cat_mask]) + self.noc_net_ret[f'image_{c_cat}'](shape_code[cat_mask])
                            scene_tar_noc_embed = self.noc_net_tar[f'pointnet_{c_cat}'](noc_points[cat_mask], pred_masks[cat_mask]) + self.noc_net_tar[f'image_{c_cat}'](shape_code[cat_mask])
                            _rets, _cads = self.joint_net[c_cat]._retrieval_inference(scene_ret_noc_embed, scene_ret_noc_embed.shape[0])
                            _rets_tmp = [_e for _e in _rets]
                            retrieved_idx_scene = [_rets_tmp.pop(0) if _f else retrieved_idx_scene[i] for i, _f in enumerate(cat_mask)]
                            cad_ids_scene = [_cads.pop(0) if _f else cad_ids_scene[i] for i, _f in enumerate(cat_mask)]
                            params_tmp = self.joint_net[c_cat]._deform_inference(scene_tar_noc_embed, _rets)
                            params_scene = [params_tmp.pop(0) if _f else params_scene[i] for i, _f in enumerate(cat_mask)]
                    else:
                        scene_noc_embeds = ret_noc_embed[scene_mask]
                        scene_tar_embeds = tar_noc_embed[scene_mask]
                        for c_cat in range(self.num_classes):
                            cat_mask = scene_classes == c_cat
                            if not cat_mask.any():
                                continue
                            scene_ret_noc_embed = scene_noc_embeds[cat_mask][...,c_cat*256:(c_cat+1)*256]
                            scene_tar_noc_embed = scene_tar_embeds[cat_mask][...,c_cat*256:(c_cat+1)*256]
                            _rets, _cads = self.joint_net[c_cat]._retrieval_inference(scene_ret_noc_embed, scene_ret_noc_embed.shape[0])
                            _rets_tmp = [_e for _e in _rets]
                            retrieved_idx_scene = [_rets_tmp.pop(0) if _f else retrieved_idx_scene[i] for i, _f in enumerate(cat_mask)]
                            cad_ids_scene = [_cads.pop(0) if _f else cad_ids_scene[i] for i, _f in enumerate(cat_mask)]
                            params_tmp = self.joint_net[c_cat]._deform_inference(scene_tar_noc_embed, _rets)
                            params_scene = [params_tmp.pop(0) if _f else params_scene[i] for i, _f in enumerate(cat_mask)]
                    cad_ids_scene.reverse()
                    params_scene.reverse()
                    retrieved_idx_scene.reverse()
                    for i, m in enumerate(scene_mask):
                        if m:
                            cad_ids[i] = cad_ids_scene.pop()
                            params[i] = params_scene.pop()
                            retrieved_idx[i] = retrieved_idx_scene.pop()
                has_alignment[[id is None for id in cad_ids]] = False
            
                pred_indices = torch.arange(pred_classes.numel(), dtype=torch.long)
                return cad_ids, pred_indices, params, retrieved_idx, torch.cat([nocs_sampled.view(_reps,3,-1).unsqueeze(1), nocs_sampled.view(_reps,3,-1).unsqueeze(1)], dim=1)
            else:
                noc_embeds = self.embed_nocs(shape_code, noc_points, pred_masks)
                for scene in set(scenes):
                    scene_mask = [scene_ == scene for scene_ in scenes]
                    scene_noc_embeds = noc_embeds[scene_mask]
                    scene_classes = pred_classes[scene_mask]

                    indices = self.indices_by_scene[scene]
                    points_by_class = {}
                    ids_by_class = {}
                    for c in scene_classes.tolist():
                        ind = indices[c]
                        if not len(ind):
                            continue
                        points_by_class[c] = self.points_by_class[c][ind]
                        ids_by_class[c] = \
                            [self.cad_ids_by_class[c][i] for i in ind]

                    cad_ids_scene = embedding_lookup(
                        scene_classes,
                        scene_noc_embeds,
                        points_by_class,
                        ids_by_class
                    )
                    cad_ids_scene.reverse()
                    for i, m in enumerate(scene_mask):
                        if m:
                            cad_ids[i] = cad_ids_scene.pop()
                has_alignment[[id is None for id in cad_ids]] = False

        pred_indices = torch.arange(pred_classes.numel(), dtype=torch.long)

        return cad_ids, pred_indices
