import torch

from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler, batch_rect_average_depth)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head_obb import BaseRoIHeadOBB
from .test_mixins import BBoxTestMixin, MaskTestMixin
from graspnetAPI.utils.utils import batch_center_depth
import numpy as np
import open3d as o3d
from mmdet.models.utils import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup


@HEADS.register_module()
class StandardRoIHeadOBB(BaseRoIHeadOBB):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_rect_grasps_hbb,
                      gt_rect_grasps,
                      gt_depths,
                      gt_scores):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            gt_rect_grasps_hbb_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_rect_grasps_hbb[i],
                    gt_scores[i], gt_rect_grasps_hbb_ignore[i],
                    gt_scores[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_rect_grasps_hbb[i],
                    gt_scores[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        img_shapes = tuple(meta['pad_shape'] for meta in img_metas)
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, score_pred, grasp_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      gt_rect_grasps,
                                                      gt_depths,
                                                      gt_scores,
                                                      self.train_cfg,
                                                      img_shapes)
            loss_bbox = self.bbox_head.loss(cls_score,
                                            score_pred,
                                            grasp_pred,
                                            *bbox_targets)

            losses.update(loss_bbox)

        return losses


    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    depth,
                    proposal_list,
                    img_metas,
                    dataset='graspnet',
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, score_pred, grasp_pred = self.bbox_head(bbox_feats)

        img_shapes = tuple(meta['pad_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if 'center_crop' in img_metas[0]:
            center_crop = tuple(meta['center_crop'] for meta in img_metas)
            center_crop_xstart = tuple(meta['center_crop_xstart'] for meta in img_metas)
            center_crop_ystart = tuple(meta['center_crop_ystart'] for meta in img_metas)
        else:
            center_crop = tuple(None for meta in img_metas)
            center_crop_xstart = tuple(None for meta in img_metas)
            center_crop_ystart = tuple(None for meta in img_metas)
        num_proposals_per_img = tuple(len(p) for p in proposal_list)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        score_pred = score_pred.split(num_proposals_per_img, 0)
        grasp_pred = grasp_pred.split(num_proposals_per_img, 0)

        det_grasp_groups = []
        for i in range(len(proposal_list)):
            det_rect_grasp_group = self.bbox_head.get_det_rect_grasp_group(rois[i],
                                                                           cls_score[i],
                                                                           score_pred[i],
                                                                           grasp_pred[i],
                                                                           dataset=dataset,
                                                                           center_crop=center_crop[i],
                                                                           center_crop_xstart=center_crop_xstart[i],
                                                                           center_crop_ystart=center_crop_ystart[i],
                                                                           img_shape=img_shapes[i],
                                                                           ori_shape=ori_shapes[i],
                                                                           scale_factor=scale_factors[i],
                                                                           rescale=True,
                                                                           cfg=self.test_cfg)
            if dataset == 'graspnet':
                if 'depth_method' not in self.test_cfg:
                    depth_method = None
                elif self.test_cfg['depth_method'] == 'batch_center_depth':
                    depth_method = batch_center_depth
                elif self.test_cfg['depth_method'] == 'batch_rect_average_depth':
                    depth_method = batch_rect_average_depth
                det_grasp_group = det_rect_grasp_group.to_grasp_group(self.test_cfg['camera'],
                                                                      depth[i].detach().cpu().numpy(),
                                                                      depth_method, ori_shapes[i])
                # det_grasp_group = self.collision_detection(det_grasp_group, depth[i].detach().cpu().numpy().squeeze(0), self.test_cfg['camera'])

            else:
                det_grasp_group = det_rect_grasp_group

            det_grasp_groups.append(det_grasp_group)
        return det_grasp_groups

    def collision_detection(self, gg, depth, camera):
        mfcdetector = ModelFreeCollisionDetector(depth, camera, voxel_size=0.008)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.001,
                                            empty_thresh=0.15)
        gg = gg[~collision_mask]
        if len(gg) == 0:
            return GraspGroup()
        return gg