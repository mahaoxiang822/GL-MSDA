from .two_stage_obb import TwoStageOBBDetector
import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, rbbox_to_hbbox_list,
                        points_to_xywhtheta_list,  xywhtheta_to_rect_grasp_group)
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmcv.runner import auto_fp16
import os
import cv2
import numpy as np
from mmdet.models.utils import (DomainAdaptationModule, RGBDDomainAdaptationModule,
                                RotatePrediction, Decoder)
from torchvision import transforms

@DETECTORS.register_module
class FasterRCNNOBBRGBDDDPretrain(BaseDetector):

    def __init__(self,
                 dump_folder,
                 rgb_backbone,
                 depth_backbone,
                 rotate_prediction,
                 train_cfg,
                 test_cfg,
                 task_type='rotate',
                 pretrained=None):
        super(FasterRCNNOBBRGBDDDPretrain, self).__init__()
        self.rgb_backbone = build_backbone(rgb_backbone)
        self.depth_backbone = build_backbone(depth_backbone)

        self.task_type = task_type
        if self.task_type == 'rotate' or self.task_type == 'crop_rotate':
            self.relative_rotation = RotatePrediction(**rotate_prediction)
        elif self.task_type == 'decode':
            self.rgb_decoder = Decoder()
            self.depth_decoder = Decoder()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dump_folder = dump_folder

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        super(FasterRCNNOBBRGBDDDPretrain, self).init_weights(pretrained)
        self.rgb_backbone.init_weights(pretrained=pretrained)
        self.depth_backbone.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        # 提取特征层的最终特征
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def generate_rgd(self, rgb, depth):
        rgds = []
        for i in range(len(rgb)):
            rg = rgb[i, :2, :, :]
            d = depth[i]
            rgd = torch.cat([rg, d], dim=0)
            rgds.append(rgd)
        rgds = torch.stack(rgds, dim=0)
        return rgds


    @auto_fp16(apply_to=('rgb', 'depth'))
    def forward(self, rgb, img_metas, depth=None, origin_depth=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(rgb=rgb, depth=depth, origin_depth=origin_depth, img_metas=img_metas, **kwargs)
        else:
            return self.forward_test(rgbs=rgb, depths=depth, origin_depths=origin_depth, img_metas=img_metas, **kwargs)


    def forward_train(self,
                      rgb,
                      depth,
                      origin_depth,
                      img_metas,
                      gt_rect_grasps,
                      gt_labels=None,
                      gt_depths=None,
                      gt_scores=None,
                      gt_object_ids=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # rgb concat ddd

        depth = depth.repeat(1, 3, 1, 1)
        x_rgb = self.rgb_backbone(rgb)
        x_depth = self.depth_backbone(depth)

        losses = dict()

        if self.task_type == 'rotate':
            rgb_rotate_angle = rgb.new_zeros(rgb.shape[0], dtype=torch.int)
            rgb_rotate_angle[:] = torch.randint(0, 4, (rgb.shape[0] - 1,))
            depth_rotate_angle = rgb.new_zeros(rgb.shape[0], dtype=torch.int)
            depth_rotate_angle[:] = torch.randint(0, 4, (rgb.shape[0] - 1,))
            rgb_crop_rotate = []
            depth_crop_rotate = []
            for i in range(len(img_metas)):
                rgb_tmp = torch.rot90(rgb[i,:,:,:], k=rgb_rotate_angle[i], dims=[1, 2])
                depth_tmp = torch.rot90(depth[i,:,:,:], k=depth_rotate_angle[i], dims=[1, 2])

                rgb_crop_rotate.append(rgb_tmp.unsqueeze(0))
                depth_crop_rotate.append(depth_tmp.unsqueeze(0))
            rgb_crop_rotate = torch.cat(rgb_crop_rotate, dim=0)
            depth_crop_rotate = torch.cat(depth_crop_rotate, dim=0)

            x_rgb_crop_rotate = self.rgb_backbone(rgb_crop_rotate)
            x_depth_crop_rotate = self.depth_backbone(depth_crop_rotate)

            loss_relative_rotation = self.relative_rotation.forward_train(x_rgb[0], x_rgb_crop_rotate[0], x_depth[0], x_depth_crop_rotate[0], rgb_rotate_angle, depth_rotate_angle)

            losses.update(loss_relative_rotation)

        elif self.task_type == 'crop_rotate':
            start_x = torch.randint(0, rgb.shape[-2] - 512, (rgb.shape[0] * 2,))
            start_y = torch.randint(0, rgb.shape[-1] - 910, (rgb.shape[0] * 2,))
            rgb_rotate_angle = rgb.new_zeros(rgb.shape[0], dtype=torch.int)
            rgb_rotate_angle[:] = torch.randint(0, 4, (rgb.shape[0] - 1,))
            depth_rotate_angle = rgb.new_zeros(rgb.shape[0], dtype=torch.int)
            depth_rotate_angle[:] = torch.randint(0, 4, (rgb.shape[0] - 1,))
            rgb_crop_rotate = []
            depth_crop_rotate = []
            for i in range(len(img_metas)):
                rgb_tmp = rgb[i, :, start_x[2 * i]:start_x[2 * i] + 512, start_y[2 * i]:start_y[2 * i] + 910]
                rgb_tmp = torch.rot90(rgb_tmp, k=rgb_rotate_angle[i], dims=[1, 2])
                depth_tmp = depth[i, :, start_x[2 * i + 1]:start_x[2 * i + 1] + 512, start_y[2 * i + 1]:start_y[2 * i + 1] + 910]
                depth_tmp = torch.rot90(depth_tmp, k=depth_rotate_angle[i], dims=[1, 2])

                rgb_crop_rotate.append(rgb_tmp.unsqueeze(0))
                depth_crop_rotate.append(depth_tmp.unsqueeze(0))
            rgb_crop_rotate = torch.cat(rgb_crop_rotate, dim=0)
            depth_crop_rotate = torch.cat(depth_crop_rotate, dim=0)

            x_rgb_crop_rotate = self.rgb_backbone(rgb_crop_rotate)
            x_depth_crop_rotate = self.depth_backbone(depth_crop_rotate)

            loss_relative_rotation = self.relative_rotation.forward_train(x_rgb[0], x_rgb_crop_rotate[0], x_depth[0],
                                                                          x_depth_crop_rotate[0], rgb_rotate_angle,
                                                                          depth_rotate_angle)

            losses.update(loss_relative_rotation)

        return losses

    def forward_test(self, rgbs, img_metas, depths=None, origin_depths=None, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(rgbs, 'rgb'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(rgbs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(rgbs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(rgbs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            if depths is not None:
                depth = depths[0]
            else:
                depth = None
            if origin_depths is not None:
                origin_depth = origin_depths[0]
            else:
                origin_depth = None
            # if gt_rect_grasps is not None:
            #     gt_rect_grasp = gt_rect_grasps[0]
            #     gt_score = gt_scores[0]
            # else:
            #     gt_rect_grasp = None
            #     gt_score = None
            return self.simple_test(rgb=rgbs[0], depth=depth, origin_depth=origin_depth,
                                    img_metas=img_metas[0], **kwargs)
        else:
            assert rgbs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{rgbs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(rgbs, img_metas, **kwargs)



    def simple_test(self,
                    rgb,
                    depth,
                    origin_depth,
                    img_metas,
                    gt_rect_grasps=None,
                    gt_scores=None,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        # dataset = 'cornell'
        dataset = 'graspnet'

        # x = self.extract_feat(rgb)

        # rgd
        # rgd = self.generate_rgd(rgb, depth)
        # x = self.extract_feat(rgd)

        # rgb + ddd
        # x_rgb = self.rgb_backbone(rgb)
        # depth = depth.repeat(1, 3, 1, 1)
        # x_depth = self.depth_backbone(depth)
        # x = []
        # for i in range(len(x_rgb)):
        #     x.append(x_rgb[i] + x_depth[i])
        # x = tuple(x)

        # rgb concat ddd
        if self.fusion_type == 'concat_every':
            depth = depth.repeat(1, 3, 1, 1)
            x_depth = self.depth_backbone(depth)
            x = self.rgb_backbone(rgb, x_depth)
        elif self.fusion_type == 'concat_final':
            depth = depth.repeat(1, 3, 1, 1)
            x_rgb = self.rgb_backbone(rgb)
            x_depth = self.depth_backbone(depth)
            x = torch.cat((x_rgb[0], x_depth[0]), dim=1)
            x = (self.fusion[0](x),)


        proposal_list = self.rpn_head.simple_test_rpn(x,
                                                      # origin_depth,
                                                      img_metas)

        results = self.roi_head.simple_test(x,
                                            origin_depth,
                                            proposal_list, img_metas, dataset=dataset, rescale=rescale)


        if dataset == 'graspnet':
            dump_folder = self.dump_folder
            eval_root = os.path.join(os.path.abspath('.'), dump_folder)
            if not os.path.exists(eval_root):
                os.makedirs(eval_root)
            for i in range(len(results)):
                sceneId = img_metas[i]['sceneId']
                annId = img_metas[i]['annId']
                camera = img_metas[i]['camera']
                eval_dir = os.path.join(eval_root, 'scene_%04d' % sceneId, camera)
                if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir)
                eval_file = os.path.join(eval_dir, "%04d.npy" %annId)
                results[i].save_npy(eval_file)



        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

