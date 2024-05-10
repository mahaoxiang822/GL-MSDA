from ..builder import PIPELINES
import os
import mmcv
import cv2
import numpy as np
import copy


@PIPELINES.register_module()
class LoadRGBDepthGraspNet(object):
    def __init__(self,
                 with_rgb=True,
                 with_depth=True,
                 with_origin_depth=False,
                 with_segment=False,
                 file_client_args=dict(backend='disk')):
        self.with_rgb = with_rgb
        self.with_depth = with_depth
        self.with_origin_depth = with_origin_depth
        self.with_segment = with_segment
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        这里得到的rgb项实际是BGR格式
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        rgb_filename = results['img_info']['rgb_filename']
        depth_filename = results['img_info']['depth_filename']
        origin_depth_filename = results['img_info']['origin_depth_filename']
        segment_filename = results['img_info']['segment_filename']

        if self.with_rgb:
            bgr_bytes = self.file_client.get(os.path.join(results['data_root'], rgb_filename))
            bgr = mmcv.imfrombytes(bgr_bytes, flag='color', channel_order='bgr')
            results['rgb'] = bgr
            if bgr is None:
                print(rgb_filename)
            results['image_fields'].append('rgb')
        if self.with_depth:
            # depth_bytes = self.file_client.get(os.path.join(results['data_root'], "scenes", depth_filename))
            depth_bytes = self.file_client.get(os.path.join(results['data_root'],  depth_filename))
            depth = mmcv.imfrombytes(depth_bytes, flag='unchanged')
            results['depth'] = depth.astype(np.float32)
            results['image_fields'].append('depth')
        if self.with_origin_depth:
            # origin_depth_bytes = self.file_client.get(os.path.join(results['data_root'], "scenes", origin_depth_filename))
            origin_depth_bytes = self.file_client.get(os.path.join(results['data_root'], origin_depth_filename))
            origin_depth = mmcv.imfrombytes(origin_depth_bytes, flag='unchanged')
            results['origin_depth'] = origin_depth.astype(np.float32)
            results['image_fields'].append('origin_depth')
        if self.with_segment:
            segment_bytes = self.file_client.get(os.path.join(results['data_root'], segment_filename))
            segment = mmcv.imfrombytes(segment_bytes, flag='unchanged')
            results['segment'] = segment
            results['image_fields'].append('segment')


        results['sceneId'] = results['img_info']['sceneId']
        results['annId'] = results['img_info']['annId']
        results['camera'] = results['img_info']['camera']
        results['domain'] = results['img_info']['domain']
        results['rgb_filename'] = rgb_filename
        results['depth_filename'] = depth_filename
        results['img_shape'] = bgr.shape
        results['ori_shape'] = bgr.shape

        return results


@PIPELINES.register_module()
class LoadAnnotationsGraspNet(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 file_client_args=dict(backend='disk')):
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_grasps(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_rect_grasps'] = ann_info['rect_grasps'].copy()
        results['gt_depths'] = ann_info['depths'].copy()
        results['gt_scores'] = ann_info['scores'].copy()
        results['gt_object_ids'] = ann_info['object_ids'].copy()
        results['rect_grasp_fields'].append('gt_rect_grasps')
        results['rect_grasp_fields'].append('gt_depths')
        results['rect_grasp_fields'].append('gt_scores')
        results['rect_grasp_fields'].append('gt_object_ids')

        # scores = results['gt_scores']
        # depths = results['gt_depths']
        # object_ids = results['gt_object_ids']
        # rect_grasps = results['gt_rect_grasps']
        # img_shape = results['img_shape']
        # center_x = (rect_grasps[:, 0] + rect_grasps[:, 2] + rect_grasps[:, 4] + rect_grasps[:, 6]) / 4
        # center_y = (rect_grasps[:, 1] + rect_grasps[:, 3] + rect_grasps[:, 5] + rect_grasps[:, 7]) / 4
        # valid_mask1 = np.logical_and(center_x < img_shape[1], center_x >= 0)
        # valid_mask2 = np.logical_and(center_y < img_shape[0], center_y >= 0)
        # valid_mask = np.logical_and(valid_mask1, valid_mask2)
        # results['gt_rect_grasps'] = rect_grasps[valid_mask]
        # results['gt_depths'] = depths[valid_mask]
        # results['gt_scores'] = scores[valid_mask]
        # results['gt_object_ids'] = object_ids[valid_mask]
        return results


    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        results = self._load_grasps(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadRGBDFromWebcamGraspNet(object):
    def __init__(self,
                 to_float32=False,
                 with_rgb=True,
                 with_depth=True,
                 with_origin_depth=True,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.with_rgb = with_rgb
        self.with_depth = with_depth
        self.with_origin_depth = with_origin_depth
        self.file_client_args = file_client_args.copy()
        self.file_client = None


    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        这里得到的rgb项实际是BGR格式
        """

        if self.with_rgb:
            rgb = results['rgb']
            if self.to_float32:
                rgb = rgb.astype(np.float32)
        if self.with_depth:
            depth = results['depth']
            if self.to_float32:
                depth = depth.astype(np.float32)

        if self.with_origin_depth:
            origin_depth = results['origin_depth']
            if self.to_float32:
                origin_depth = origin_depth.astype(np.float32)

        results['rgb'] = rgb
        results['depth'] = depth
        results['origin_depth'] = origin_depth
        results['rgb_filename'] = None
        results['depth_filename'] = None
        results['ori_filename'] = None
        results['img_shape'] = rgb.shape
        results['ori_shape'] = rgb.shape
        results['image_fields'] = ['rgb', 'depth', 'origin_depth']
        results['rect_grasp_fields'] = []
        return results
