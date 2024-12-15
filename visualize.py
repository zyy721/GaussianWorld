import os
offscreen = False
if os.environ.get('DISP', 'f') == 'f':
    from pyvirtualdisplay import Display
    display = Display(visible=False, size=(2560, 1440))
    display.start()
    offscreen = True
# from xvfbwrapper import Xvfb
# vdisplay = Xvfb(width=1920, height=1080)
# vdisplay.start()

from mayavi import mlab
import mayavi
mlab.options.offscreen = offscreen
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.iou_eval import IOUEvalBatch
from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging.logger import MMLogger
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def plot_opa_hist(opas, save_name):
    plt.cla(); plt.clf()
    plt.hist(opas, range=(0, 1), bins=20)
    plt.savefig(save_name)
    plt.cla(); plt.clf()

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def draw(
    voxels=None,          # semantic occupancy predictions
    gauss=None,           # semantic gaussians
    vox_origin=None,
    voxel_size=0.2,  # voxel size in the real world
    sem=False,
    save_path=None
):
    if voxels is not None:
        w, h, z = voxels.shape
        # grid = grid.astype(np.int)
        # voxels[98:102, 95:105, 8:10] = 0

        # Compute the voxels coordinates
        grid_coords = get_grid_coords(
            [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
        ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    elif gauss is not None:
        grid_coords = gauss[:, [1,0,2,3]]

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 17)
    ]
    print('occ num:', len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            # transparent=True,
            # vmin=1,
            # vmax=40, # 16
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            # transparent=True,
            vmin=0,
            vmax=16, # 16
        )

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    if sem:
        colors = np.array(
            [
                [  0,   0,   0, 255],       # others
                [255, 120,  50, 255],       # barrier              orange
                [255, 192, 203, 255],       # bicycle              pink
                [255, 255,   0, 255],       # bus                  yellow
                [  0, 150, 245, 255],       # car                  blue
                [  0, 255, 255, 255],       # construction_vehicle cyan
                [255, 127,   0, 255],       # motorcycle           dark orange
                [255,   0,   0, 255],       # pedestrian           red
                [255, 240, 150, 255],       # traffic_cone         light yellow
                [135,  60,   0, 255],       # trailer              brown
                [160,  32, 240, 255],       # truck                purple                
                [255,   0, 255, 255],       # driveable_surface    dark pink
                # [175,   0,  75, 255],       # other_flat           dark red
                [139, 137, 137, 255],
                [ 75,   0,  75, 255],       # sidewalk             dard purple
                [150, 240,  80, 255],       # terrain              light green          
                [230, 230, 250, 255],       # manmade              white
                [  0, 175,   0, 255],       # vegetation           green
                # [  0, 255, 127, 255],       # ego car              dark cyan
                # [255,  99,  71, 255],       # ego car
                # [  0, 191, 255, 255]        # ego car
            ]
        ).astype(np.uint8)
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    scene = figure.scene
    if True:
        scene.camera.position = [  -35.08337438, 7.5131739, 16.71378558]
        scene.camera.focal_point = [  -34.21734897, 7.5131739, 16.21378558]
        scene.camera.view_angle = 40.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [0.01, 300.]
        scene.camera.compute_view_plane_normal()
        scene.render()
    else:
        scene.camera.position = [118.7195754824976, 118.70290907014409, 120.11124225247899]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [114.42016931210819, 320.9039783052695]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.azimuth(-5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(5)
        scene.render()
        scene.camera.azimuth(-5)
        scene.render()
        scene.camera.position = [-138.7379881436844, -0.008333206176756428, 99.5084646673331]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [104.37185230017721, 252.84608651497263]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-114.65804807470022, -0.008333206176756668, 82.48137575398867]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [75.17498702830105, 222.91192666552377]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-94.75727115818437, -0.008333206176756867, 68.40940144543957]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.0, 0.0, 1.0]
        scene.camera.clipping_range = [51.04534630774225, 198.1729515833347]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.6463156430702276, -6.454925414290924e-18, 0.7630701733934554]
        scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.position = [-107.15500034628069, -0.008333206176756742, 92.16667026873841]
        scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
        scene.camera.view_angle = 30.0
        scene.camera.view_up = [0.6463156430702277, -6.4549254142909245e-18, 0.7630701733934555]
        scene.camera.clipping_range = [78.84362692774403, 218.2948716014858]
        scene.camera.compute_view_plane_normal()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.elevation(5)
        scene.camera.orthogonalize_view_up()
        scene.render()
        scene.camera.elevation(-5)
        mlab.pitch(-8)
        mlab.move(up=15)
        scene.camera.orthogonalize_view_up()
        scene.render()

    if offscreen:
        mlab.savefig(save_path, size=(2560, 1440))
    else:
        mlab.show()
    # mlab.savefig(save_path)
    mlab.close()

def pass_print(*args, **kwargs):
    pass

def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0

def main(args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    cfg.val_dataset_config.scene_name = args.scene_name

    # init DDP
    distributed = True
    world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    gpu = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend="nccl", init_method=f"env://", 
        world_size=world_size, rank=rank
    )
    # dist.barrier()
    torch.cuda.set_device(gpu)

    if not is_main_process():
        import builtins
        builtins.print = pass_print

    # configure logger
    if is_main_process():
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='bevworld', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    print('done ddp model')

    # build dataloader
    from dataset import build_dataloader
    train_dataset_loader, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            dist=distributed,
        )

    amp = cfg.get('amp', True)
    from loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda()
    batch_iou = len(cfg.model.encoder.return_layer_idx)
    CalMeanIou_sem = IOUEvalBatch(n_classes=18, bs=batch_iou, device=torch.device('cpu'), ignore=[0], is_distributed=distributed)
    CalMeanIou_geo = IOUEvalBatch(n_classes=2, bs=batch_iou, device=torch.device('cpu'), ignore=[], is_distributed=distributed)
    
    # resume and load
    if args.load_from:
        cfg.load_from = args.load_from
    print('work dir: ', args.work_dir)
    if cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
        
    if cfg.val_dataset_config.scene_name is None:
        save_dir = os.path.join(args.work_dir)
    else:
        save_dir = os.path.join(args.work_dir, cfg.val_dataset_config.scene_name)
    os.makedirs(save_dir, exist_ok=True)

    my_model.eval()
    CalMeanIou_sem.reset()
    CalMeanIou_geo.reset()
    loss_record = LossRecord(loss_func=loss_func)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, label) = data

            F = imgs.shape[1]
            history_anchor = None
            for i in range(F):
                with torch.cuda.amp.autocast(enabled=amp):
                    result_dict = my_model(imgs=imgs[:, i], metas=metas[0][i], label=label[:, i:i+1], history_anchor=history_anchor)
                if args.stream_test:
                    history_anchor = result_dict['history_anchor']
                loss, loss_dict = loss_func(result_dict)
            
                loss_record.update(loss=loss.item(), loss_dict=loss_dict)
                voxel_predict = result_dict['ce_input'].argmax(dim=1).long()
                voxel_label = result_dict['ce_label'].long()
                iou_predict = ((voxel_predict > 0) & (voxel_predict < 17)).long()
                iou_label = ((voxel_label > 0) & (voxel_label < 17)).long()
                CalMeanIou_sem.addBatch(voxel_predict, voxel_label)
                CalMeanIou_geo.addBatch(iou_predict, iou_label)

                frame_idx = i

                # vis occ
                if args.vis_occ:
                    voxel_predict = torch.argmax(result_dict['ce_input'][-1], dim=0).long()
                    voxel_label = result_dict['ce_label'][-1].long()
                    voxel_origin = cfg.pc_range[:3]
                    resolution = 0.4
                    voxel_predict[voxel_predict==0] = 17
                    to_vis = voxel_predict.clone().cpu().numpy()
                    save_path = os.path.join(save_dir, f'occ_frame_{frame_idx}.png')
                    draw(to_vis, 
                        None,
                        voxel_origin, 
                        [resolution] * 3, 
                        sem=True,
                        save_path=save_path)
                    # voxel_label[voxel_label==0] = 17
                    # to_vis = voxel_label.clone().cpu().numpy()
                    # save_path = os.path.join(save_dir, f'occ_gt_frame_{frame_idx}.png')
                    # draw(to_vis, 
                    #     None,
                    #     voxel_origin, 
                    #     [resolution] * 3, 
                    #     sem=True,
                    #     save_path=save_path)
            
            if i_iter_val % 1 == 0 and is_main_process():
                loss_info = loss_record.loss_info()
                logger.info('[EVAL] Iter %5d/%d   '%(i_iter_val, len(val_dataset_loader)) + loss_info)
                # loss_record.reset()

    val_iou_sem = CalMeanIou_sem.getIoU()
    val_iou_geo = CalMeanIou_geo.getIoU()
    info_sem = [[float('{:.4f}'.format(iou)) for iou in val_iou_sem[i, 1:17].mean(-1, keepdim=True).tolist()] for i in range(val_iou_sem.shape[0])]
    info_geo = [float('{:.4f}'.format(iou)) for iou in val_iou_geo[:, 1].tolist()]

    logger.info(val_iou_sem.cpu().tolist())
    logger.info(f'Current val iou of sem is {info_sem}')
    logger.info(f'Current val iou of geo is {info_geo}')
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_occ.py')
    parser.add_argument('--work-dir', type=str, default='./work_dir/tpv_occ')
    parser.add_argument('--load-from', type=str, default=None)
    parser.add_argument('--scene-name', type=str, default=None)
    parser.add_argument('--vis_occ', action='store_true')
    parser.add_argument('--stream-test', action='store_true')

    args, _ = parser.parse_known_args()
    main(args)