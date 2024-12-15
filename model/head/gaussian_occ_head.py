import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from mmengine import MODELS
from mmengine.model import BaseModule
from .localagg.local_aggregate import LocalAggregator
from ..encoder.gaussian_encoder.utils import \
    cartesian, safe_sigmoid, GaussianPrediction, get_rotation_matrix


@MODELS.register_module()
class GaussianOccHead(BaseModule):
    def __init__(
        self,
        empty_label=17,
        num_classes=18,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=0.4),
        with_empty=False,
        empty_args=dict(),
        pc_range=[],
        scale_range=[],
        include_opa=True,
        semantics_activation='softmax'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.aggregator = LocalAggregator(**cuda_kwargs)
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10, requires_grad=True)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.empty_label = empty_label
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.include_opa = include_opa
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = self.num_classes if not with_empty else self.num_classes - 1
        self.semantics_activation = semantics_activation
        xyz = self.get_meshgrid(pc_range, [cuda_kwargs['H'], cuda_kwargs['W'], cuda_kwargs['D']], cuda_kwargs['grid_size'])
        self.register_buffer('gt_xyz', torch.tensor(xyz)[None])
    
    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3
    
    def anchor2gaussian(self, anchor):
        xyz = cartesian(anchor, self.pc_range)
        gs_scales = safe_sigmoid(anchor[..., 3:6])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        rot = anchor[..., 6: 10]
        opas = safe_sigmoid(anchor[..., 10: (10 + int(self.include_opa))])
        semantics = anchor[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)
        
        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            opacities=opas,
            semantics=semantics
        )
        return gaussian
    
    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            # if 'kitti' in self.dataset_type:
            #     opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            # else:
            #     opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)
            opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)
            empty_mean = self.empty_mean.clone().repeat([means.shape[0], 1, 1])
            means = torch.cat([means, empty_mean], dim=1)
            empty_scale = self.empty_scale.clone().repeat([scales.shape[0], 1, 1])
            scales = torch.cat([scales, empty_scale], dim=1)
            empty_rot = self.empty_rot.clone().repeat([rotations.shape[0], 1, 1])
            rotations = torch.cat([rotations, empty_rot], dim=1)
            empty_sem = self.empty_sem.clone().repeat([opacities.shape[0], 1, 1])
            empty_sem[..., self.empty_label] += self.empty_scalar
            opacities = torch.cat([opacities, empty_sem], dim=1)
            empty_opa = self.empty_opa.clone().repeat([origi_opa.shape[0], 1, 1])
            origi_opa = torch.cat([origi_opa, empty_opa], dim=1)

        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = Cov.cpu().inverse().cuda() # b, g, 3, 3
        return means, origi_opa, opacities, scales, CovInv
    
    def prepare_gt_xyz(self, tensor):
        B, G, C = tensor.shape
        gt_xyz = self.gt_xyz.repeat([B, 1, 1, 1, 1]).to(tensor.dtype)
        return gt_xyz

    def forward(self, anchors, label, output_dict):
        B, F, G, _ = anchors.shape
        assert B==1
        anchors = anchors.flatten(0, 1)
        gaussians = self.anchor2gaussian(anchors)
        means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussians)

        gt_xyz = self.prepare_gt_xyz(anchors)        # bf, x, y, z, 3
        sampled_xyz = gt_xyz.flatten(1, 3).float()
        origi_opa = origi_opa.flatten(1, 2)
        
        semantics = []
        for i in range(len(sampled_xyz)):
            semantic = self.aggregator(
                sampled_xyz[i:(i+1)], 
                means[i:(i+1)], 
                origi_opa[i:(i+1)],
                opacities[i:(i+1)],
                scales[i:(i+1)],
                CovInv[i:(i+1)]) # n, c
            semantics.append(semantic)
        semantics = torch.stack(semantics, dim=0).transpose(1, 2)
        spatial_shape = label.shape[2:]
        
        output_dict.update({
            'ce_input': semantics.unflatten(-1, spatial_shape), # F, 17, 200, 200, 16
            'ce_label': label.squeeze(0),                       # F, 200, 200, 16
            # 'gaussians': gaussians,
            # 'anchors': anchors,
        })
        return output_dict

