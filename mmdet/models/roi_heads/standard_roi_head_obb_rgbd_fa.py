import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler,
                        batch_rect_average_depth,
                        xyxydepth2roi)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head_obb import BaseRoIHeadOBB
from .test_mixins import BBoxTestMixin, MaskTestMixin
from graspnetAPI.utils.utils import batch_center_depth
import numpy as np
from mmdet.models.utils import ModelFreeCollisionDetector


@HEADS.register_module()
class StandardRoIHeadFA(BaseRoIHeadOBB):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_fa(self, calGP=False, fa_classes=12, gp_path=None):
        self.calGP = calGP
        self.fa_classes = fa_classes
        self.gp_path = gp_path
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7)
        # self.GPS = nn.Parameter(torch.zeros(2048, fa_classes, device='cpu'), requires_grad=False)
        # self.GPT = nn.Parameter(torch.zeros(2048, fa_classes, device='cpu'), requires_grad=False)
        self.GPS = torch.zeros(2048, fa_classes, device='cpu').cuda()
        self.GPT = torch.zeros(2048, fa_classes, device='cpu').cuda()
        self.source_nums = torch.zeros(fa_classes)
        self.target_nums = torch.zeros(fa_classes)
        self.loaded = False

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
                      origin_depths,
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

        if not self.loaded:
            self.GPS = torch.load(self.gp_path + 'gps.pth', map_location='cpu')
            self.GPT = torch.load(self.gp_path + 'gpt.pth', map_location='cpu')
            print('************************************************')
            print('we get GPS and GOT from ' + self.gp_path)
            self.loaded = True
            device = x[0].device
            self.GPS = self.GPS.to(device)
            self.GPT = self.GPT.to(device)

        self.GPS = self.GPS.detach()
        self.GPT = self.GPT.detach()

        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            gt_rect_grasps_hbb_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            domains = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_rect_grasps_hbb[i],
                    gt_scores[i], gt_rect_grasps_hbb_ignore[i],
                    gt_scores[i], img_metas[i]['domain'])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_rect_grasps_hbb[i],
                    # gt_depths[i],
                    gt_scores[i],
                    img_metas[i]['domain'],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
                domains.append(img_metas[i]['domain'])

        losses = dict()
        # bbox head forward and loss
        img_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        if self.with_bbox:
            bbox_nums = []
            for res in sampling_results:
                bbox_nums.append(len(res.bboxes))
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, score_pred, grasp_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      origin_depths,
                                                      gt_rect_grasps,
                                                      gt_depths,
                                                      gt_scores,
                                                      self.train_cfg,
                                                      img_shapes,
                                                      domains)
            features = self.avg_pool(bbox_feats)
            features = features.view(features.size(0), -1)

            domain_masks = bbox_targets[6].type(torch.bool)

            target_scores = F.softmax(cls_score[~domain_masks], dim=1)
            pos_inds_target = (target_scores[:, 0] > 0.2).type(torch.bool)
            source_labels = bbox_targets[0][domain_masks]
            pos_inds_source = (source_labels == 0).type(torch.bool)

            target_grasp_pred = grasp_pred[~domain_masks][pos_inds_target]
            target_sin2theta = target_grasp_pred[:,4]
            target_cos2theta = target_grasp_pred[:,5]
            target_theta_pred = torch.atan2(target_sin2theta, target_cos2theta) / 2
            target_features = features[~domain_masks][pos_inds_target]

            grasp_label = bbox_targets[4]
            source_grasp_label = grasp_label[domain_masks][pos_inds_source]
            source_sin2theta = source_grasp_label[:, 4]
            source_cos2theta = source_grasp_label[:, 5]
            source_theta_label = torch.atan2(source_sin2theta, source_cos2theta) / 2
            source_features = features[domain_masks][pos_inds_source]

            pi = torch.acos(torch.zeros((1))).item() * 2

            target_index = target_theta_pred + pi / 2
            target_index = target_index / (pi / self.fa_classes)
            target_index = target_index.long().squeeze()
            nums = torch.zeros(self.fa_classes)
            PT = torch.zeros_like(self.GPT)
            for myi in range(len(target_index)):
                ind = target_index[myi]
                if ind == 12:
                    ind -= 1
                nums[ind] = nums[ind] + 1
                PT[:, ind] = PT[:, ind] + target_features[myi]

            for myi in range(self.fa_classes):
                if nums[myi] > 0:
                    PT[:, myi] = PT[:, myi] / nums[myi]
                    similarity = F.cosine_similarity(PT[:, myi], self.GPT[:, myi], dim=0)
                    similarity = (similarity + 1) / 2
                    if myi == 0:
                        # losses['target_similarity'] = similarity.detach()
                        print('sim', similarity, 'PT:', torch.sum(torch.abs(PT[:, myi])), 'GPT:',
                              torch.sum(torch.abs(self.GPT[:, myi])))
                    similarity = similarity * 0.001
                    self.GPT[:, myi] = similarity.detach() * PT[:, myi] + (1 - similarity.detach()) * self.GPT[:, myi]


            # update GPT from around
            # for myi in range(self.fa_classes):
            #     if nums[myi] > 0:
            #         PT[:, myi] = PT[:, myi] / nums[myi]
            #
            # for myi in range(self.fa_classes):
            #     if nums[myi] > 0:
            #         similarity_mid = F.cosine_similarity(PT[:, myi], self.GPT[:, myi], dim=0)
            #         similarity_mid = (similarity_mid + 1) / 2 * 0.001 * 0.5
            #         similarity_mid.detach_()
            #         myi_mins1 = myi - 1
            #         if myi_mins1 == 0:
            #             myi_mins1 = self.fa_classes - 1
            #         similarity_mins1 = F.cosine_similarity(PT[:, myi_mins1], self.GPT[:, myi], dim=0)
            #         similarity_mins1 = (similarity_mins1 + 1) / 2 * 0.001 * 0.25
            #         similarity_mins1.detach_()
            #         myi_add1 = myi + 1
            #         if myi_add1 == self.fa_classes:
            #             myi_add1 = 0
            #         similarity_add1 = F.cosine_similarity(PT[:, myi_add1], self.GPT[:, myi], dim=0)
            #         similarity_add1 = (similarity_add1 + 1) / 2 * 0.001 * 0.25
            #         similarity_add1.detach_()
            #         self.GPT[:, myi] = similarity_mid * PT[:, myi] + similarity_mins1 * PT[:, myi_mins1] + \
            #                            similarity_add1 * PT[:, myi_add1] + \
            #                            (1 - similarity_add1 - similarity_mins1 - similarity_mid) * self.GPT[:, myi]




            source_index = source_theta_label + pi / 2
            source_index = source_index / (pi / self.fa_classes)
            source_index = source_index.long().squeeze()
            nums = torch.zeros(self.fa_classes)
            PS = torch.zeros_like(self.GPS)
            for myi in range(len(source_index)):
                ind = source_index[myi]
                if ind == 12:
                    ind -= 1
                nums[ind] = nums[ind] + 1
                PS[:, ind] = PS[:, ind] + source_features[myi]
            for myi in range(self.fa_classes):
                if nums[myi] > 0:
                    PS[:, myi] = PS[:, myi] / nums[myi]
                    similarity = F.cosine_similarity(PS[:, myi], self.GPS[:, myi], dim=0)
                    similarity = (similarity + 1) / 2
                    if myi == 0:
                        # losses['source_similarity'] = similarity.detach()
                        print('sim',similarity,'PS:',torch.sum(torch.abs(PS[:,myi])),'GPS:',torch.sum(torch.abs(self.GPS[:,myi])))
                    similarity = similarity * 0.001
                    self.GPS[:, myi] = similarity.detach() * PS[:, myi] + (1 - similarity.detach()) * self.GPS[:, myi]

            # update GPS from around
            # for myi in range(self.fa_classes):
            #     if nums[myi] > 0:
            #         PS[:, myi] = PS[:, myi] / nums[myi]
            #
            # for myi in range(self.fa_classes):
            #     if nums[myi] > 0:
            #         similarity_mid = F.cosine_similarity(PS[:, myi], self.GPS[:, myi], dim=0)
            #         similarity_mid = (similarity_mid + 1) / 2 * 0.001 * 0.5
            #         similarity_mid.detach_()
            #         myi_mins1 = myi - 1
            #         if myi_mins1 == 0:
            #             myi_mins1 = self.fa_classes - 1
            #         similarity_mins1 = F.cosine_similarity(PS[:, myi_mins1], self.GPS[:, myi], dim=0)
            #         similarity_mins1 = (similarity_mins1 + 1) / 2 * 0.001 * 0.25
            #         similarity_mins1.detach_()
            #         myi_add1 = myi + 1
            #         if myi_add1 == self.fa_classes:
            #             myi_add1 = 0
            #         similarity_add1 = F.cosine_similarity(PS[:, myi_add1], self.GPS[:, myi], dim=0)
            #         similarity_add1 = (similarity_add1 + 1) / 2 * 0.001 * 0.25
            #         similarity_add1.detach_()
            #         self.GPS[:, myi] = similarity_mid * PS[:, myi] + similarity_mins1 * PS[:, myi_mins1] + \
            #                            similarity_add1 * PS[:, myi_add1] + \
            #                            (1 - similarity_add1 - similarity_mins1 - similarity_mid) * self.GPS[:, myi]


            loss_GPS = self.GPS.clone()
            loss_GPT = self.GPT.clone()
            fa_loss = dict()
            fa_loss['fa_loss'] =500 * F.mse_loss(loss_GPS, loss_GPT)
            losses.update(fa_loss)

            # loss_GPS = self.GPS.clone()
            # loss_GPT = self.GPT.clone()
            # loss_GPS_back = self.GPS.clone()
            # loss_GPS_back = torch.cat((loss_GPS_back[:,-1].unsqueeze(1), loss_GPS_back[:,:-1]), dim=1)
            # loss_GPS_front = self.GPS.clone()
            # loss_GPS_front = torch.cat((loss_GPS_front[:,1:], loss_GPS_front[:,0].unsqueeze(1)), dim=1)
            # fa_loss = dict()
            # fa_loss['fa_loss'] = 1000 * F.mse_loss(loss_GPS, loss_GPT) * 0.6 + 1000 * F.mse_loss(loss_GPS_back, loss_GPT) * 0.2  \
            #                      + 1000 * F.mse_loss(loss_GPS_front, loss_GPT) * 0.2
            # print('mid: ', torch.norm(loss_GPS[:, 1]))
            # print('back: ', torch.norm(loss_GPS_back[:, 2]))
            # print('front: ', torch.norm(loss_GPS_front[:, 0]))
            # losses.update(fa_loss)

            loss_bbox = self.bbox_head.loss(cls_score,
                                            score_pred,
                                            grasp_pred,
                                            *bbox_targets)

            losses.update(loss_bbox)

        return losses, bbox_feats, bbox_targets[-1], bbox_nums


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
                    origin_depths,
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
        # proposals = xyxydepth2roi(proposal_list).split(num_proposals_per_img, 0)

        det_grasp_groups = []
        # rbboxes = []
        # labels = []
        for i in range(len(proposal_list)):
            det_rect_grasp_group = self.bbox_head.get_det_rect_grasp_group(rois[i],
                                                                           cls_score[i],
                                                                           score_pred[i],
                                                                           grasp_pred[i],
                                                                           origin_depths[i].squeeze(),
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
                                                                      origin_depths[i].detach().cpu().numpy(),
                                                                      depth_method, ori_shapes[i])
                # det_grasp_group = self.collision_detection(det_grasp_group, origin_depths[i].detach().cpu().numpy().squeeze(0),
                #                                            self.test_cfg['camera'])

                # det_grasp_group.translation[:, 2] -= 0.005
                # det_grasp_group.depths = 0.015 * np.ones((len(det_grasp_group)))
                # det_grasp_group.translations = det_grasp_group.translations - det_grasp_group.rotation_matrices[:, :, 0] * \
                #                                self.test_cfg['refine_approach_dist']
            else:
                det_grasp_group = det_rect_grasp_group

            det_grasp_groups.append(det_grasp_group)
            # rbboxes.append(rbbox)
            # labels.append(label)

        return det_grasp_groups # , rbboxes, labels

    def collision_detection(self, gg, depth, camera):
        mfcdetector = ModelFreeCollisionDetector(depth, camera, voxel_size=0.008)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.001,
                                            empty_thresh=0.15)
        gg = gg[~collision_mask]
        return gg
