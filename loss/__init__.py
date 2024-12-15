from mmengine.registry import Registry
GPD_LOSS = Registry('gpd_loss')

from .multi_loss import MultiLoss
from .ce_loss import CELoss
from .lovasz_loss import LovaszLoss
from .l2_loss import L2Loss
from .l1_loss import L1Loss
from .sem_geo_loss import Sem_Scal_Loss, Geo_Scal_Loss
from .focal_loss import FocalLoss