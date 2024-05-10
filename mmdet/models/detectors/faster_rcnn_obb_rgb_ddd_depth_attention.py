from .two_stage_obb import TwoStageOBBDetector
import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, rbbox_to_hbbox_list,
                        points_to_xywhtheta_list,  xywhtheta_to_rect_grasp_group)
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmcv.runner import auto_fp16
import os
from mmdet.models.utils import (SpatialAttentionLayer, CrossModalMultiHeadAttention,
                                CrossModalMultiHeadAttentionK, CrossModalMultiHeadAttention_origin)
import numpy as np
import cv2
from sklearn.decomposition import PCA


@DETECTORS.register_module
class FasterRCNNOBBRGBDDDDepthAttention(BaseDetector):

    def __init__(self,
                 dump_folder,
                 rgb_backbone,
                 depth_backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 fusion_type=None,
                 num_head=(8, 8, 8),
                 head_dim=(32, 64, 128),
                 sample_nums=(3, 3, 3),
                 with_position_encoding=(True, True, True),
                 position_encoding_type='sin'):
        super(FasterRCNNOBBRGBDDDDepthAttention, self).__init__()
        self.rgb_backbone = build_backbone(rgb_backbone)
        self.depth_backbone = build_backbone(depth_backbone)

        self.fusion = nn.ModuleList()
        self.fusion_type = fusion_type
        for i in range(rgb_backbone['num_stages']):
            if fusion_type == 'spatial_d2rgb' or fusion_type == 'spatial_rgb2d' or fusion_type == 'spatial_d2rgb_mask':
                self.fusion.append(SpatialAttentionLayer(in_planes=2,
                                                       out_planes=1,
                                                       kernel_size=7,
                                                       stride=1,
                                                       padding=3,
                                                       bn=False,
                                                       relu=False))
            elif fusion_type == 'gate' or fusion_type == 'sum':
                pass
            elif fusion_type == 'cross_d2rgb':
                self.fusion.append(CrossModalMultiHeadAttention(in_channels=64 * 2 ** (i+2),
                                                                num_head=1,
                                                                ratio=2))
            elif fusion_type == 'cross_d2rgb_k':
                self.fusion.append(CrossModalMultiHeadAttentionK(in_channels=64 * 2 ** (i+2),
                                                                 num_head=num_head[i],
                                                                 head_dim=head_dim[i],
                                                                 sample_nums=sample_nums[i],
                                                                 with_position_encoding=with_position_encoding[i],
                                                                 position_encoding_type=position_encoding_type))
            elif fusion_type == 'cross_d2rgb_origin':
                self.fusion.append(CrossModalMultiHeadAttention_origin(in_channels=64 * 2 ** (i+2),
                                                                       num_head=1,
                                                                       ratio=2))


        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

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
        super(FasterRCNNOBBRGBDDDDepthAttention, self).init_weights(pretrained)
        self.rgb_backbone.init_weights(pretrained=pretrained)
        # self.depth_backbone.init_weights(pretrained=pretrained)
        self.depth_backbone.init_weights(pretrained=None)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)
        for i in range(len(self.fusion)):
            self.fusion[i].init_weights()

    def extract_feat(self, img):
        # 提取特征层的最终特征
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat_sum(self, rgb, depth):
        # rgb + ddd
        depth = depth.repeat(1, 3, 1, 1)
        x_depth = self.depth_backbone(depth)
        if self.rgb_backbone.deep_stem:
            x_rgb = self.rgb_backbone.stem(rgb)
        else:
            x_rgb = self.rgb_backbone.conv1(rgb)
            x_rgb = self.rgb_backbone.norm1(x_rgb)
            x_rgb = self.rgb_backbone.relu(x_rgb)
        x_rgb = self.rgb_backbone.maxpool(x_rgb)
        outs = []
        for i, layer_name in enumerate(self.rgb_backbone.res_layers):
            rgb_res_layer = getattr(self.rgb_backbone, layer_name)
            x_rgb = rgb_res_layer(x_rgb)
            x_rgb = x_rgb + x_depth[i]
            if i in self.rgb_backbone.out_indices:
                outs.append(x_rgb)
        return tuple(outs)


    def extract_feat_cross_d2rgb_k(self, rgb, depth, scene_id=None, ann_id=None, img_metas=None):
        depth = depth.repeat(1, 3, 1, 1)
        x_depth = self.depth_backbone(depth)

        if self.rgb_backbone.deep_stem:
            x_rgb = self.rgb_backbone.stem(rgb)
        else:
            x_rgb = self.rgb_backbone.conv1(rgb)
            x_rgb = self.rgb_backbone.norm1(x_rgb)
            x_rgb = self.rgb_backbone.relu(x_rgb)
            x_rgb = self.rgb_backbone.maxpool(x_rgb)
        outs = []
        for i, layer_name in enumerate(self.rgb_backbone.res_layers):
            rgb_res_layer = getattr(self.rgb_backbone, layer_name)
            x_rgb = rgb_res_layer(x_rgb)
            x_rgb, _ = self.fusion[i](key=x_depth[i], query=x_rgb)
            if i in self.rgb_backbone.out_indices:
                outs.append(x_rgb)
        return tuple(outs)




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
        gt_rect_grasps_hbb = rbbox_to_hbbox_list(gt_rect_grasps)
        gt_rect_grasps_xywhtheta = points_to_xywhtheta_list(gt_rect_grasps)

        # x = self.extract_feat(rgb)

        # rgd
        # rgd = self.generate_rgd(rgb, depth)
        # x = self.extract_feat(rgd) # x得到的是一个元组，对于resnet，可包含多个conv层的结果


        if self.fusion_type == 'cross_d2rgb_k':
            x = self.extract_feat_cross_d2rgb_k(rgb, depth)
        elif self.fusion_type == 'sum':
            x = self.extract_feat_sum(rgb, depth)


        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_rect_grasps_hbb,
                gt_scores,
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

        roi_losses = self.roi_head.forward_train(x,
                                                 origin_depth,
                                                 img_metas,
                                                 proposal_list,
                                                 gt_rect_grasps_hbb,
                                                 gt_rect_grasps_xywhtheta,
                                                 gt_depths,
                                                 gt_scores,
                                                 **kwargs)
        losses.update(roi_losses)

        if torch.isnan(losses['loss_grasp']):
            import pdb
            pdb.set_trace()

        # torch.cuda.empty_cache()

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
                # print(img.size())
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


        if self.fusion_type == 'spatial_d2rgb':
            x = self.extract_feat_spatial_d2rgb(rgb, depth, img_metas)
        elif self.fusion_type == 'spatial_rgb2d':
            x = self.extract_feat_spatial_rgb2d(rgb, depth)
        elif self.fusion_type == 'spatial_d2rgb_mask':
            x = self.extract_feat_spatial_d2rgb_mask(rgb, depth, origin_depth)
        elif self.fusion_type == 'gate':
            x = self.extract_feat_gate(rgb, depth)
        elif self.fusion_type == 'cross_d2rgb' or self.fusion_type == 'cross_d2rgb_origin':
            x = self.extract_feat_cross_d2rgb(rgb, depth)
        elif self.fusion_type == 'cross_d2rgb_k':
            x = self.extract_feat_cross_d2rgb_k(rgb, depth, scene_id=img_metas[0]['sceneId'], ann_id=img_metas[0]['annId'])
        elif self.fusion_type == 'sum':
            x = self.extract_feat_sum(rgb, depth)

        proposal_list = self.rpn_head.simple_test_rpn(x,
                                                      # origin_depth,
                                                      img_metas)

        results= self.roi_head.simple_test(x,
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

