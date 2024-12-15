
import numpy as np
import torch
from torch.utils import data
from . import OPENOCC_DATAWRAPPER
from dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, ImageAug3D


# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

@OPENOCC_DATAWRAPPER.register_module()
class NuScenes_Scene_Occ_DatasetWrapper_Stream(data.Dataset):
    def __init__(self, in_dataset, final_dim=[256, 704], resize_lim=[0.45, 0.55], flip=False, phase='train'):
        self.dataset = in_dataset
        self.phase = phase
        if phase == 'train':
            transforms = [
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, flip=flip, is_train=True),
                PhotoMetricDistortionMultiViewImage(),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        else:
            transforms = [
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, flip=False, is_train=False),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        imgs, metas, occ = data

        # deal with img augmentation
        F, N, H, W, C = imgs.shape
        imgs_dict = {'img': imgs.reshape(F*N, H, W, C)}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = np.stack([img.transpose(2, 0, 1) for img in imgs], axis=0)
        FN, C, H, W = imgs.shape
        imgs = imgs.reshape(F, N, C, H, W)

        img_aug_matrix = None
        if imgs_dict.get('img_aug_matrix'):
            img_aug_matrix = np.stack(imgs_dict['img_aug_matrix'], axis=0).reshape(F, N, 4, 4)
        for i in range(F):
            metas[i][0]['img_shape'] = imgs_dict['img_shape'][6*i:6*(i+1)]
            if img_aug_matrix is not None:
                metas[i][0]['img_aug_matrix'] = img_aug_matrix[i:i+1]
        
        data_tuple = (imgs, metas, occ)

        return data_tuple