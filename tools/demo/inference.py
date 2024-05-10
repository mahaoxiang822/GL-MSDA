import numpy as np
from mmdet.apis import init_detector, inference_planer_grasp_rgbd
import cv2
import os
import open3d as o3d
from mmcv.parallel import MMDataParallel
import time
import torch
import argparse


class PlanerGrasp:
    def __init__(self, config_path, checkpoint_path):
        device = 'cuda:0'
        # init a detector
        self.model = init_detector(config_path, checkpoint_path, device=device)
        # self.model = MMDataParallel(model, device_ids=[0])

    def detect(self, rgb, depth, scene_id, ann_id, camera):
        gg = inference_planer_grasp_rgbd(self.model, rgb, depth, scene_id, ann_id, camera)[0]
        return gg

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--scene_id', type=int, help='scene id')
    parser.add_argument('--ann_id', type=int, help='ann id')
    parser.add_argument('--camera', help='camera')
    parser.add_argument('--source_domain', help='source_domain')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    root = "/home/qrrr/mmdetection_grasp/"
    args = parse_args()

    data_dir = os.path.join(root, "data", "planer_graspnet")
    rgb_path = os.path.join(data_dir, "scenes", "scene_%04d" % args.scene_id, args.camera, "rgb",
                            "%04d.png" % args.ann_id)
    depth_path = os.path.join(data_dir, "depths", "scene_%04d" % args.scene_id, args.camera,
                              "depth_fill_hole_bilateral_outlier",
                              "%04d.png" % args.ann_id)
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if args.camera == 'realsense' and args.source_domain == 'Sim-S':
        config_source = '/home/qrrr/mmdetection_grasp/configs/sim_to_real/demo_source_realsense.py'
        checkpoint_source = '/home/qrrr/mmdetection_grasp/work_dirs/sim_to_real/rgb_ddd_concat_final_random_sim/20221005/epoch_38.pth'
        config_sim2real = '/home/qrrr/mmdetection_grasp/configs/sim_to_real/demo_sims2realsense.py'
        checkpoint_sim2real = '/home/qrrr/mmdetection_grasp/work_dirs/sim_to_real/rgb_ddd_concat_final_sim2real/2022101602/epoch_49.pth'
    elif args.camera == 'kinect' and args.source_domain == 'Sim-S':
        config_source = '/home/qrrr/mmdetection_grasp/configs/sim_to_real/demo_source_kinect.py'
        checkpoint_source = '/home/qrrr/mmdetection_grasp/work_dirs/sim_to_real/rgb_ddd_concat_final_random_sim/20221109/epoch_42.pth'
        config_sim2real = '/home/qrrr/mmdetection_grasp/configs/sim_to_real/demo_sims2kinect.py'
        checkpoint_sim2real = '/home/qrrr/mmdetection_grasp/work_dirs/sim_to_real/rgb_ddd_concat_final_sim2real/2022111901/epoch_53.pth'
    elif args.camera == 'realsense' and args.source_domain == 'Sim-B':
        config_source = '/home/qrrr/mmdetection_grasp/configs/sim_to_real/demo_source_realsense.py'
        checkpoint_source = '/home/qrrr/mmdetection_grasp/work_dirs/sim_to_real/rgb_ddd_concat_final_random_sim/20221005/epoch_38.pth'
        config_sim2real = '/home/qrrr/mmdetection_grasp/configs/sim_to_real/demo_sims2realsense.py'
        checkpoint_sim2real = '/home/qrrr/mmdetection_grasp/work_dirs/sim_to_real/rgb_ddd_concat_final_sim2real/2022111702/epoch_36.pth'
    elif args.camera == 'kinect' and args.source_domain == 'Sim-B':
        config_source = '/home/qrrr/mmdetection_grasp/configs/sim_to_real/demo_source_kinect.py'
        checkpoint_source = '/home/qrrr/mmdetection_grasp/work_dirs/sim_to_real/rgb_ddd_concat_final_random_sim/20221109/epoch_42.pth'
        config_sim2real = '/home/qrrr/mmdetection_grasp/configs/sim_to_real/demo_sims2kinect.py'
        checkpoint_sim2real = '/home/qrrr/mmdetection_grasp/work_dirs/sim_to_real/rgb_ddd_concat_final_sim2real/2022111301/epoch_39.pth'


    pg = PlanerGrasp(config_source, checkpoint_source)
    gg = pg.detect([rgb], [depth], [args.scene_id], [args.ann_id], [args.camera])
    np.save(os.path.join(root, "source.npy"), gg.grasp_group_array)

    pg = PlanerGrasp(config_sim2real, checkpoint_sim2real)
    gg = pg.detect([rgb], [depth], [args.scene_id], [args.ann_id], [args.camera])
    np.save(os.path.join(root, "sim2real.npy"), gg.grasp_group_array)



