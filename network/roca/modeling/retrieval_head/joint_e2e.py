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
    get_symmetric
)

from pytorch3d.loss import chamfer_distance


Tensor = torch.Tensor
TensorByClass = Dict[int, Tensor]
IDByClass = Dict[int, List[Tuple[str, str]]]
RetrievalResult = Tuple[List[Tuple[str, str]], Tensor]


class JointE2EHead(nn.Module):
    def __init__(self, cfg, shape_code_size: int, margin: float = 10.):
        super().__init__()
        self.has_cads = False
        self.mode = cfg.MODEL.RETRIEVAL_MODE
        self.shape_code_size = shape_code_size
        self.is_voxel = cfg.INPUT.CAD_TYPE == 'voxel'

        # NOTE: Make them embeddings for the learned model
        self.wild_points_by_class: Optional[TensorByClass] = None
        self.wild_ids_by_class: Optional[IDByClass] = None

        self.baseline = cfg.MODEL.RETRIEVAL_BASELINE
        if self.baseline:
            return

        self.loss = nn.TripletMarginLoss(margin=margin)

        if '_' in self.mode:
            self.cad_mode, self.noc_mode = self.mode.split('_')
        else:
            self.cad_mode = self.noc_mode = self.mode

        if self.cad_mode == 'pointnet':
            assert not self.is_voxel, 'Inconsistent CAD modality'
            self.cad_net = PointNet()
        elif self.cad_mode in ('joint', "joint+e2e"):
            self.cad_net = JointNet(cfg)
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
                'pointnet': PointNet(fc_out=True),
                'image': self.make_image_mlp(),
                'comp_lat': nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU(True),
                    nn.Linear(1024, self.cad_net.embedding_dim)
                )
            })
            self.noc_net_tar = nn.ModuleDict({
                'pointnet': PointNet(fc_out=True),
                'image': self.make_image_mlp(),
                'comp_lat': nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU(True),
                    nn.Linear(1024, self.cad_net.embedding_dim)
                )
            })
        elif self.noc_mode == 'joint+resnet+image+comp':
            resnet = ResNetEncoder()
            self.noc_net_ret = nn.ModuleDict({
                'resnet': resnet,
                'image': self.make_image_mlp(),
                'comp': ResNetDecoder(relu_in=True, feats=resnet.feats)
            })
            self.noc_net_tar = nn.ModuleDict({
                'resnet': resnet,
                'image': self.make_image_mlp(),
                'comp': ResNetDecoder(relu_in=True, feats=resnet.feats)
            })
            self.comp_loss = nn.BCELoss()
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
                'image': self.make_image_mlp()
            })
        elif self.noc_mode in ('resnet+image+comp', 'resnet+image+fullcomp'):
            resnet = ResNetEncoder()
            self.noc_net = nn.ModuleDict({
                'resnet': resnet,
                'image': self.make_image_mlp(),
                'comp': ResNetDecoder(relu_in=True, feats=resnet.feats)
            })
            self.comp_loss = nn.BCELoss()
        else:
            raise ValueError('Unknown noc mode {}'.format(self.noc_mode))

    def make_image_mlp(self, relu_out: bool = True) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.shape_code_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.cad_net.embedding_dim),
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
                # ret_noc_embed = self.noc_net_ret['pointnet'](noc_points, masks) + self.noc_net_ret['image'](shape_code)
                # tar_noc_embed = self.noc_net_tar['pointnet'](noc_points, masks) + self.noc_net_tar['image'](shape_code)
                ret_noc_embed = self.noc_net_ret['comp_lat'](torch.cat([self.noc_net_ret['pointnet'](noc_points, masks), self.noc_net_ret['image'](shape_code)], dim=-1))
                tar_noc_embed = self.noc_net_tar['comp_lat'](torch.cat([self.noc_net_tar['pointnet'](noc_points, masks), self.noc_net_tar['image'](shape_code)], dim=-1))
                ret_pos_embed, ret_neg_embed, tar_pos_embed, tar_neg_embed = self.cad_net._embed_trip(pos_cads, neg_cads)
                losses['loss_triplet'] = (self.loss(ret_noc_embed, ret_pos_embed, ret_neg_embed) + self.loss(tar_noc_embed, tar_pos_embed, tar_neg_embed))
            elif self.noc_mode in ('joint+resnet+image+comp', 'joint+resnet+image+fullcomp'):
                ret_noc_embed, ret_comp, tar_noc_embed, tar_comp = self.embed_nocs(
                    shape_code=shape_code,
                    noc_points=noc_points,
                    mask=masks
                )
                losses['loss_noc_comp'] = self.comp_loss(
                    noc_comp, pos_cads.to(dtype=noc_comp.dtype)
                )
                ret_pos_embed, ret_neg_embed, tar_pos_embed, tar_neg_embed = self.cad_net._embed_trip(pos_cads, neg_cads)
                losses['loss_triplet'] = (self.loss(ret_noc_embed, ret_pos_embed, ret_neg_embed) + self.loss(tar_noc_embed, tar_pos_embed, tar_neg_embed))
            elif self.cad_mode == "joint+e2e":
                ret_noc_embed = self.noc_net_ret['comp_lat'](torch.cat([self.noc_net_ret['pointnet'](noc_points, masks), self.noc_net_ret['image'](shape_code)], dim=-1))
                tar_noc_embed = self.noc_net_tar['comp_lat'](torch.cat([self.noc_net_tar['pointnet'](noc_points, masks), self.noc_net_tar['image'](shape_code)], dim=-1))
                candies = self.cad_net.get_candidates(ret_noc_embed.detach(), noc_points.shape[0])
                out_pcs = self.cad_net._deform(tar_noc_embed, candies)
                cd_loss, _ = chamfer_distance(out_pcs, pos_cads.flatten(2), batch_reduction=None)
                fitting_loss = torch.mean(cd_loss)
                reflected_pc = get_symmetric(out_pcs)
                symmetric_loss, _ = chamfer_distance(out_pcs, reflected_pc)
                fitting_loss += symmetric_loss
                losses['loss_fitting'] = fitting_loss
                losses['loss_embed'] = self.cad_net._retrieval(ret_noc_embed, candies, fitting_loss)
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
            return self.cad_net(cad_points)
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
        # NOTE: assume cad embeddings instead of cad points
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
            if self.noc_mode == 'joint':
                _reps = noc_points.shape[0]
                nocs_sampled = noc_points.view(_reps, 3, -1)
                # ret_noc_embed = self.noc_net_ret['pointnet'](noc_points, pred_masks) + self.noc_net_ret['image'](shape_code)
                # tar_noc_embed = self.noc_net_tar['pointnet'](noc_points, pred_masks) + self.noc_net_tar['image'](shape_code)
                ret_noc_embed = self.noc_net_ret['comp_lat'](torch.cat([self.noc_net_ret['pointnet'](noc_points, pred_masks), self.noc_net_ret['image'](shape_code)], dim=-1))
                tar_noc_embed = self.noc_net_tar['comp_lat'](torch.cat([self.noc_net_tar['pointnet'](noc_points, pred_masks), self.noc_net_tar['image'](shape_code)], dim=-1))
                for scene in set(scenes):
                    scene_mask = [scene_ == scene for scene_ in scenes]
                    scene_noc_embeds = ret_noc_embed[scene_mask]
                    scene_tar_embeds = tar_noc_embed[scene_mask]
                    scene_classes = pred_classes[scene_mask]

                    retrieved_idx_scene, cad_ids_scene = self.cad_net._retrieval_inference(scene_noc_embeds, scene_noc_embeds.shape[0])
                    params_scene = self.cad_net._deform_inference(scene_tar_embeds, scene_noc_embeds.shape[0], retrieved_idx_scene, scenes=scenes, has_alignment=has_alignment, pred_classes=pred_classes)
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
                return cad_ids, pred_indices, params, retrieved_idx, torch.cat([noc_points.view(_reps, 3, -1).unsqueeze(1), noc_points.view(_reps, 3, -1).unsqueeze(1)], dim=1)
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
