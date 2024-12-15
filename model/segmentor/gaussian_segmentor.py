import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG


@MODELS.register_module()
class GaussianSegmentor(BaseModule):

    def __init__(
        self,
        backbone=None,
        neck=None,
        lifter=None,
        encoder=None,
        future_decoder=None,
        head=None, 
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if backbone is not None:
            try:
                self.backbone = MODELS.build(backbone)
            except:
                self.backbone = MODELS_SEG.build(backbone)
        if neck is not None:
            try:
                self.neck = MODELS.build(neck)
            except:
                self.neck = MODELS_SEG.build(neck)
        if lifter is not None:
            self.lifter = MODELS.build(lifter)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        if future_decoder is not None:
            self.future_decoder = MODELS.build(future_decoder)
        if head is not None:
            self.head = MODELS.build(head)

    def extract_img_feat(self, imgs):
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = self.neck(img_feats_backbone)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
    def obtain_anchor(self, imgs, metas):
        B, F, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B*F, N, C, H, W)
        mlvl_img_feats = self.extract_img_feat(imgs)
        anchor, instance_feature = self.lifter(mlvl_img_feats)    # b, g, c
        anchor = self.encoder(anchor, instance_feature, mlvl_img_feats, metas) # b, g, c
        return anchor
    
    def forward(
        self,
        imgs=None,
        metas=None,
        label=None,
        test_mode=False,
        **kwargs,
    ):
        B, F, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'

        anchor = self.obtain_anchor(imgs, metas)
        BF, G, C = anchor.shape
        anchor = anchor.reshape(B, F, G, C)
        if hasattr(self, 'future_decoder'):
            output_dict = self.future_decoder(anchor, metas)
            anchor_predict = output_dict.pop('anchor')
        else:
            anchor_predict = anchor
            output_dict = dict()
        output_dict = self.head(
            anchors=anchor_predict, 
            label=label, 
            output_dict=output_dict)

        return output_dict
    
    def forward_test(self,
                        imgs=None,
                        metas=None,
                        label=None,
                        test_mode=True,
                        **kwargs,
        ):
        B, F, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'
        
        anchor = self.obtain_anchor(imgs, metas)
        BF, G, C = anchor.shape
        anchor = anchor.reshape(B, F, G, C)

        anchor_predict = anchor
        output_dict = dict()
        output_dict = self.head(
            anchors=anchor_predict, 
            label=label, 
            output_dict=output_dict, 
            test_mode=test_mode)

        return output_dict