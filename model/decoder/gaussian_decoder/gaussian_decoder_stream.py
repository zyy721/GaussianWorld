import torch, numpy as np, torch.nn as nn
import spconv.pytorch as spconv
from mmengine import MODELS
from mmengine.model import BaseModule
from functools import partial

from ...encoder.gaussian_encoder.utils import cartesian, safe_inverse_sigmoid, LOGIT_MAX


@MODELS.register_module()
class GaussianDecoderStream(BaseModule):

    def __init__(
        self, 
        pc_range,
        num_anchor,
        embed_dims=128,
        anchor_prior_kwargs = None,
        init_cfg = None
    ):
        super().__init__(init_cfg)
        self.get_xyz = partial(cartesian, pc_range=pc_range)
        self.num_anchor = num_anchor
        self.embed_dims = embed_dims
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float), False)
        self.register_buffer('anchor_prior', self.prepare_anchor_prior(**anchor_prior_kwargs), False)
        self.register_buffer('instance_feature_prior', torch.zeros([1, embed_dims]), False)

    def prepare_anchor_prior(self, anchor_resolution, include_opa, semantic_dim, temporal_feat_dim):
        xs = torch.linspace(0, 1.0, anchor_resolution[0])[:, None, None].expand(*anchor_resolution)
        ys = torch.linspace(0, 1.0, anchor_resolution[1])[None, :, None].expand(*anchor_resolution)
        zs = torch.linspace(0, 1.0, anchor_resolution[2])[None, None, :].expand(*anchor_resolution)
        xyz = torch.stack([xs, ys, zs], dim=-1).flatten(0, 2) # XYZ, 3
        xyz = safe_inverse_sigmoid(xyz)
        num_anchors = xyz.shape[0]

        scale = torch.rand_like(xyz)
        scale = safe_inverse_sigmoid(scale)
        rots = torch.zeros(num_anchors, 4)
        rots[:, 0] = 1.0
        opacity = safe_inverse_sigmoid(0.1 * torch.ones(num_anchors, int(include_opa)))
        semantic = torch.randn(num_anchors, semantic_dim)
        temporal_feat = torch.randn(num_anchors, temporal_feat_dim, dtype=torch.float)

        anchors = torch.cat([xyz, scale, rots, opacity, semantic, temporal_feat], dim=-1)
        return anchors

    def warp_anchor(self, anchors, metas, instance_features=None):
        # anchors: bs, num_frames, g, c
        # instance_features: bs, num_framses, g, c

        # warp anchors from previous to current frame
        lidar2global = torch.tensor(metas[0]['lidar2global'], dtype=anchors.dtype, device=anchors.device)
        B, F, N, _ = anchors.shape
        xyz = self.get_xyz(anchors) # bs, f, n, 3
        prev2cur = torch.matmul(torch.linalg.inv(lidar2global[F]), lidar2global[:F])[None, :, None] # bs, f, 1, 4, 4
        new_xyz = torch.matmul(prev2cur, torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)[..., None])[..., :3, 0]     # bs, f, g, 3

        # filter out anchors beyond the boundary
        new_xyz = (new_xyz - self.pc_range[:3]) / (self.pc_range[3:] - self.pc_range[:3])
        valid_mask = ((new_xyz > 1-LOGIT_MAX) & (new_xyz < LOGIT_MAX)).all(-1)
        anchors_warp = torch.cat([safe_inverse_sigmoid(new_xyz), anchors[..., 3:]], dim=-1)     # bs, f, g, c
        anchors_warp = anchors_warp[valid_mask]
        instance_features_warp = instance_features[valid_mask]

        # fill new areas with prior gaussians
        anchor_prior = self.anchor_prior
        with torch.no_grad():
            prior_xyz = self.get_xyz(anchor_prior) # g, 3
            cur_xyz = torch.matmul(torch.linalg.inv(prev2cur[0, -1]), torch.concat([prior_xyz, torch.ones_like(prior_xyz[..., :1])], -1)[..., None])[..., :3, 0]
            mask = (cur_xyz[..., 0] < self.pc_range[0]) | (cur_xyz[..., 0] > self.pc_range[3]) | \
                (cur_xyz[..., 1] < self.pc_range[1]) | (cur_xyz[..., 1] > self.pc_range[4])
            fill_anchor = anchor_prior[mask]
            num_tofill = self.num_anchor - valid_mask[:, -1].detach().sum().item()
            metas[0]['fill_num'] = num_tofill
            if num_tofill > 0:
                if fill_anchor.shape[0] > 0:
                    fill_anchor = fill_anchor[np.random.choice(fill_anchor.shape[0], num_tofill)]
                else:
                    fill_anchor = anchor_prior[np.random.choice(anchor_prior.shape[1], num_tofill)]

                anchors_warp = torch.cat([anchors_warp, fill_anchor], dim=0)

                instance_features_fill = self.instance_feature_prior.clone().repeat([num_tofill, 1])
                instance_features_warp = torch.cat([instance_features_warp, instance_features_fill], dim=0)

        return anchors_warp.unsqueeze(0), instance_features_warp.unsqueeze(0)

    def forward(self, anchors, metas, instance_features=None):
        ### warp anchors from previous to current frame
        anchors, instance_features = self.warp_anchor(anchors, metas, instance_features)
        
        return anchors, instance_features
