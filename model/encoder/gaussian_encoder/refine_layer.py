from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale

from .utils import linear_relu_ln, safe_sigmoid, GaussianPrediction, LOGIT_MAX
import torch, torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class SparseGaussian3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        semantic_dim=0,
        with_empty=True,
        include_opa=True,
        temporal_scale=0.1,
        dynamic_scale=1.0,
    ):
        super(SparseGaussian3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = semantic_dim - 1 if with_empty else semantic_dim
        self.with_empty = with_empty
        self.output_dim = 10 + int(include_opa) + semantic_dim
        self.temporal_scale = temporal_scale
        self.dynamic_scale = dynamic_scale
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float), False)
        self.register_buffer('unit_xyz', torch.tensor(unit_xyz, dtype=torch.float), False)
        self.register_buffer('scale_range', torch.tensor(scale_range, dtype=torch.float), False)
        
        self.output_layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2, embed_dims*2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))
    
    def safe_inverse_sigmoid(self, x, range):
        x = (x - range[:3]) / (range[3:] - range[:3])
        x = torch.clamp(x, 1 - LOGIT_MAX, LOGIT_MAX)
        # x = torch.clamp(x, 1 - LOGIT_MAX, LOGIT_MAX).detach() + x - x.detach()
        return torch.log(x / (1 - x))

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        fill_num: int,
        refine_temporal_semantic: bool=False,
    ):
        output = self.output_layers(torch.cat([instance_feature, anchor_embed], dim=-1))
        if fill_num >= 0:
            tmp_num = anchor.shape[1] - fill_num
            if refine_temporal_semantic:
                output = torch.cat([output[:, :tmp_num] * self.temporal_scale, output[:, tmp_num:]], dim=1)
            else:
                semantic = F.softplus(anchor[:, :tmp_num, self.semantic_start : (self.semantic_start+self.semantic_dim)])
                semantic_weight = semantic.max(-1, keepdim=True)[0] * (torch.argmax(semantic, -1, keepdim=True) < 11) / 10 * self.dynamic_scale
                output = torch.cat([output[:, :tmp_num] * semantic_weight, output[:, tmp_num:]], dim=1)
                output[:, :tmp_num, 3:] = output[:, :tmp_num, 3:] * 0

        delta_xyz = (2 * safe_sigmoid(output[..., :3]) - 1) * self.unit_xyz
        delta_rot = torch.nn.functional.normalize(output[..., 6:10] + torch.tensor([1e-5, 0, 0, 0], dtype=output.dtype, device=output.device)[None, None], dim=-1)
        delta_scale = output[..., 3:6]
        delta_sem = output[..., 10:]
        # if fill_num >= 0 and not refine_temporal_semantic:
        #     delta_rot = torch.cat([delta_rot[:, :tmp_num] * 0 + torch.tensor([1, 0, 0, 0], dtype=delta_rot.dtype, device=delta_rot.device)[None, None], 
        #                            delta_rot[:, tmp_num:]], dim=1)
        #     delta_scale = torch.cat([delta_scale[:, :tmp_num] * 0, delta_scale[:, tmp_num:]], dim=1)
        #     delta_sem = torch.cat([delta_sem[:, :tmp_num] * 0, delta_sem[:, tmp_num:]], dim=1)
        
        # refine xyz
        xyz = safe_sigmoid(anchor[..., :3]) * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
        xyz = xyz + delta_xyz
        xyz = self.safe_inverse_sigmoid(xyz, self.pc_range)

        # refine scale
        scale = anchor[..., 3:6] + delta_scale

        # refine rot
        rot = anchor[..., 6:10]
        rot = torch.stack([
            delta_rot[..., 0] * rot[..., 0] - delta_rot[..., 1] * rot[..., 1] - delta_rot[..., 2] * rot[..., 2] - delta_rot[..., 3] * rot[..., 3],
            delta_rot[..., 0] * rot[..., 1] + delta_rot[..., 1] * rot[..., 0] + delta_rot[..., 2] * rot[..., 3] - delta_rot[..., 3] * rot[..., 2],
            delta_rot[..., 0] * rot[..., 2] + delta_rot[..., 2] * rot[..., 0] - delta_rot[..., 1] * rot[..., 3] + delta_rot[..., 3] * rot[..., 1],
            delta_rot[..., 0] * rot[..., 3] + delta_rot[..., 3] * rot[..., 0] + delta_rot[..., 1] * rot[..., 2] - delta_rot[..., 2] * rot[..., 1],
        ], dim=-1)
        rot = torch.nn.functional.normalize(rot, dim=-1)

        # refine feature like opa \ temporal feat \ semantic
        feat = anchor[..., 10:] + delta_sem

        anchor_refine = torch.cat([xyz, scale, rot, feat], dim=-1)

        return anchor_refine