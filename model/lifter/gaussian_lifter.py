import numpy as np
import torch
from torch import nn
import numpy as np
from mmengine import MODELS
from mmengine.model import BaseModule
from ..encoder.gaussian_encoder.utils import safe_inverse_sigmoid


@MODELS.register_module()
class GaussianLifter(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_anchor=25600,
        anchor_grad=True,
        semantic_dim=0,
        include_opa=True,
        temporal_feat_dim=128,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        xyz = safe_inverse_sigmoid(xyz)
        scale = torch.rand_like(xyz)
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1
        opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, int(include_opa)), dtype=torch.float))
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        temporal_feat = torch.randn(num_anchor, temporal_feat_dim, dtype=torch.float)

        anchor = torch.cat([xyz, scale, rots, opacity, semantic, temporal_feat], dim=-1)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)

    def forward(self, mlvl_img_feats):
        batch_size = mlvl_img_feats[0].shape[0]
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        return anchor