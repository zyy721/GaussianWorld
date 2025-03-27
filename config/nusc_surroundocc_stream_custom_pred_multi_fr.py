optimizer_wrapper = dict(
    optimizer = dict(
        type='AdamW',
        # lr=4e-4,
        lr=1e-5,

        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),}
    ),
)
grad_max_norm = 35
amp = False

seed = 1
print_freq = 50
scheduler = 'cosine'
warmup_iters = 1000
warmup_lr_init = 1e-6
eval_freq = 100
# eval_freq = 1
save_freq = 1


# max_epochs = 100
p_frame_schedule = [[0.2, 5], [0.1, 5], [0.05, 5], [0.033, 5], [0., 10]]
frame_schedule = [[  5, 5], [ 10, 5], [  20, 5], [   30, 5], [38, 10]]
# load_from = 'out/ckpt_base.pth'
load_from = 'out/ckpt_stream.pth'

# find_unused_parameters = False
find_unused_parameters = True

track_running_stats = True
ignore_label = 0
empty_idx = 17   # 0 noise, 1~16 objects, 17 empty
cls_dims = 18

pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
scale_range = [0.08, 0.64]
image_size = [864, 1600]
resize_lim = [1.0, 1.0]
flip = True
# num_frames = 30
# num_frames = 3
num_frames = 6
times = 5

num_frames_past = 2

temporal_feat_dim = 0
temporal_scale = 1.0
dynamic_scale = 1.0
num_learnable_pts = 6
learnable_scale = 5
scale_multiplier = 5
num_encoder = 4
num_refine_temporal = 3
return_layer_idx = [2, 3]

_dim_ = 128
num_cams = 6
num_heads = 4
num_levels = 4
drop_out = 0.1
num_anchor = 25600
semantics_activation = 'softplus'
include_opa = True
wempty = True
freeze_perception = True

anchor_encoder = dict(
    type='SparseGaussian3DEncoder',
    embed_dims=_dim_, 
    include_opa=include_opa,
    semantic_dim=cls_dims,
)

ffn = dict(
    type="AsymmetricFFN",
    in_channels=_dim_ * 2,
    embed_dims=_dim_,
    feedforward_channels=_dim_ * 4,
    pre_norm=dict(type="LN"),
    num_fcs=2,
    ffn_drop=drop_out,
    act_cfg=dict(type="ReLU", inplace=True),
)

deformable_layer = dict(
    type='DeformableFeatureAggregation',
    embed_dims=_dim_,
    num_groups=num_heads,
    num_levels=num_levels,
    num_cams=num_cams,
    attn_drop=0.15,
    use_deformable_func=True,
    use_camera_embed=True,
    residual_mode="cat",
    kps_generator=dict(
        type="SparseGaussian3DKeyPointsGenerator",
        embed_dims=_dim_,
        num_learnable_pts=num_learnable_pts,
        learnable_scale=learnable_scale,
        fix_scale=[
            [0, 0, 0],
            [0.45, 0, 0],
            [-0.45, 0, 0],
            [0, 0.45, 0],
            [0, -0.45, 0],
            [0, 0, 0.45],
            [0, 0, -0.45],
        ],
        pc_range=pc_range,
        scale_range=scale_range),
)

refine_layer = dict(
    type='SparseGaussian3DRefinementModule',
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    unit_xyz=[4.0, 4.0, 1.0],
    semantic_dim=cls_dims,
    with_empty=wempty,
    include_opa=include_opa,
    temporal_scale=temporal_scale,
    dynamic_scale=dynamic_scale,
)

spconv_layer=dict(
    type='SparseConv3DBlock',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    use_out_proj=True,
    grid_size=[0.5]*3,
    kernel_size=[5, 5, 5],
    stride=[1, 1, 1],
    padding=[2, 2, 2],
    dilation=[1, 1, 1],
)

future_decoder=dict(
    type='GaussianDecoderStream',
    pc_range=pc_range,
    num_anchor=num_anchor,
    embed_dims=_dim_,
    anchor_prior_kwargs=dict(
        anchor_resolution=[200, 200, 16],
        include_opa=include_opa,
        temporal_feat_dim=temporal_feat_dim,
        semantic_dim=cls_dims),
)

model = dict(
    # type='GaussianSegmentorStream',
    # type='GaussianSegmentorStreamCustom',
    type='GaussianSegmentorStreamCustomPredMultiFr',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp = True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
        init_cfg=dict(
          type='Pretrained',
          checkpoint='pretrain/r101_dcn_fcos3d_pretrain.pth'),
    ),
    neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=1,
        out_channels=_dim_,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    lifter=dict(
        type='GaussianLifter',
        embed_dims=_dim_,
        num_anchor=num_anchor,
        anchor_grad=False,
        include_opa=include_opa,
        temporal_feat_dim=temporal_feat_dim,
        semantic_dim=cls_dims),
    encoder=dict(
        type='GaussianEncoder',
        return_layer_idx=return_layer_idx,
        num_encoder=num_encoder,
        num_refine_temporal=num_refine_temporal,
        anchor_encoder=anchor_encoder,
        norm_layer=dict(type="LN", normalized_shape=_dim_),
        ffn=ffn,
        deformable_model=deformable_layer,
        refine_layer=refine_layer,
        spconv_layer=spconv_layer,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "spconv",
            # "norm",
            "refine",
        ] * (num_encoder)),

    pred=dict(
        type='GaussianPred',
        return_layer_idx=return_layer_idx,
        num_encoder=num_encoder,
        num_refine_temporal=num_refine_temporal,
        anchor_encoder=anchor_encoder,
        norm_layer=dict(type="LN", normalized_shape=_dim_),
        ffn=ffn,
        deformable_model=deformable_layer,
        refine_layer=refine_layer,
        spconv_layer=spconv_layer,
        operation_order=[
            # "deformable",
            # "ffn",
            # "norm",
            "spconv",
            # "norm",
            "refine",
        ] * (num_encoder)),

    future_decoder=future_decoder,
    head=dict(
        type='GaussianOccHead',
        empty_label=empty_idx,
        num_classes=cls_dims,
        cuda_kwargs=dict(
            scale_multiplier=scale_multiplier,
            H=200, W=200, D=16,
            pc_min=[-50.0, -50.0, -5.0],
            grid_size=0.5),
        with_empty=wempty,
        empty_args=dict(
                mean=[0, 0, -1.0],
                scale=[1000, 1000, 1000]),
        pc_range=pc_range,
        scale_range=scale_range,
        include_opa=include_opa,
        semantics_activation=semantics_activation
    ))


loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='CELoss',
            weight=10.0,
            cls_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],
            ignore_label=ignore_label,
            input_dict={
                'ce_input': 'ce_input',
                'ce_label': 'ce_label'}),
        dict(
            type='LovaszLoss',
            weight=1.0,
            empty_idx=empty_idx,
            ignore_label=ignore_label,
            input_dict={
                'lovasz_input': 'ce_input',
                'lovasz_label': 'ce_label'}),
    ]
)

data_path = 'data/nuscenes/'

train_dataset_config = dict(
    # type='NuScenes_Scene_SurroundOcc_Dataset_Stream',
    type='NuScenes_Scene_SurroundOcc_Dataset_Stream_Custom',
    data_path = data_path,
    num_frames = num_frames,
    imageset = 'data/nuscenes_temporal_infos_train.pkl',
    phase='train',
    times=times,
)

val_dataset_config = dict(
    # type='NuScenes_Scene_SurroundOcc_Dataset_Stream',
    type='NuScenes_Scene_SurroundOcc_Dataset_Stream_Traverse_Custom',
    data_path = data_path,
    num_frames = num_frames,
    imageset = 'data/nuscenes_temporal_infos_val.pkl',
    phase='val',
    times=times,
)

train_wrapper_config = dict(
    type='NuScenes_Scene_Occ_DatasetWrapper_Stream',
    final_dim = image_size,
    resize_lim = resize_lim,
    flip = flip,
    phase='train', 
)

val_wrapper_config = dict(
    type='NuScenes_Scene_Occ_DatasetWrapper_Stream',
    final_dim = image_size,
    resize_lim = resize_lim,
    flip = flip,
    phase='val', 
)

train_loader_config = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)
    
val_loader_config = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)