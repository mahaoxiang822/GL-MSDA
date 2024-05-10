import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.core import (multi_apply, xywhthetadepth_to_rect_grasp_group, xywhtheta_to_points_graspnet,
                        multiclass_poly_nms_8_points, points_to_xywhtheta_graspnet,
                        multiclass_poly_nms_8_points_graspnet, hbbox_to_xywhthetaz,
                        xyxydepth_to_xywhthetadepth,  build_bbox_coder)
from mmdet.models.utils import ModelFreeCollisionDetector

@HEADS.register_module
class BBoxHeadGraspNetDepth(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYZWHSinCosGraspCoder',
                     target_means=[0., 0., 0., 0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.1, 0.2, 0.2, 1., 1.]),
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_score=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_grasp=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)                 ):
        super(BBoxHeadGraspNetDepth, self).__init__()
        self.with_avg_pool = with_avg_pool


        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_score = build_loss(loss_score)
        self.loss_grasp = build_loss(loss_grasp)


        in_channels = self.in_channels # light-head-rcnn:10
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area # light-head-rcnn:10*7*7
        self.fc_cls = nn.Linear(in_channels, num_classes + 1)
        out_dim_reg_score = 1 if reg_class_agnostic else 1 * num_classes
        self.fc_reg_score = nn.Linear(in_channels, 1)
        # 会将特征图展开，这里的in_channels = self.inchannels * roi_feat_area
        out_dim_reg_grasp = 7 if reg_class_agnostic else 7 * num_classes
        self.fc_reg_grasp = nn.Linear(in_channels, out_dim_reg_grasp)
        self.debug_imgs = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

        nn.init.normal_(self.fc_reg_score.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg_score.bias, 0)

        nn.init.normal_(self.fc_reg_grasp.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg_grasp.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        score_pred = self.fc_reg_score(x)
        grasp_pred = self.fc_reg_grasp(x)
        return cls_score, score_pred, grasp_pred

    def _get_target_single(self,
                           origin_depth,
                           pos_bboxes,
                           neg_bboxes,
                           gt_rect_grasps,
                           gt_depths,
                           pos_assigned_gt_inds,
                           pos_gt_scores,
                           img_shape,
                           cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        score_targets = pos_bboxes.new_zeros(num_samples, 1)
        score_weights = pos_bboxes.new_zeros(num_samples, 1)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 7)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 7)

        pos_gt_rect_grasps = gt_rect_grasps[pos_assigned_gt_inds]  # 带方向
        pos_gt_depths = gt_depths[pos_assigned_gt_inds].view(-1, 1)
        pos_gt_rect_grasps = torch.cat((pos_gt_rect_grasps, pos_gt_depths), dim=-1)

        pos_rect_grasps_xywhthetaz = hbbox_to_xywhthetaz(pos_bboxes,
                                                                origin_depth.squeeze(),
                                                                img_shape)

        if num_pos > 0:
            labels[:num_pos] = 0
            pos_gt_scores = pos_gt_scores.view(-1, 1)[:num_pos]
            score_targets[:num_pos] = pos_gt_scores
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            score_weights[:num_pos, :] = 1
            pos_bbox_targets = self.bbox_coder.encode(
                pos_rect_grasps_xywhthetaz, pos_gt_rect_grasps)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, score_targets, score_weights, bbox_targets, bbox_weights


    def get_targets(self,
                    sampling_results,
                    origin_depths,
                    gt_rect_grasps,
                    gt_depths,
                    gt_scores,
                    rcnn_train_cfg,
                    img_shapes,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_gt_scores = [res.pos_gt_labels for res in sampling_results]
        origin_depths_list = [origin_depths[i] for i in range(len(origin_depths))]

        labels, label_weights, \
        score_targets, score_weights,\
        bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            origin_depths_list,
            pos_bboxes_list,
            neg_bboxes_list,
            gt_rect_grasps,
            gt_depths,
            pos_assigned_gt_inds,
            pos_gt_scores,
            img_shapes,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            score_weights = torch.cat(score_weights, 0)
            score_targets = torch.cat(score_targets, 0)
        return labels, label_weights, score_targets, score_weights, bbox_targets, bbox_weights



    @force_fp32(apply_to=('cls_score', 'score_pred', 'bbox_pred'))
    def loss(self,
             cls_score,
             score_pred,
             bbox_pred,
             labels,
             label_weights,
             score_targets,
             score_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 7)[pos_inds.type(torch.bool)]
                pos_score_pred = score_pred.view(score_pred.size(0), 1)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    7)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]
                pos_score_pred = score_pred.view(score_pred.size(0), -1, 1)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]
            losses['loss_grasp'] = self.loss_grasp(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
            losses['loss_score'] = self.loss_score(
                pos_score_pred,
                score_targets[pos_inds],
                score_weights[pos_inds],
                avg_factor=score_targets.size(0),
                reduction_override=reduction_override)


        return losses

    def get_det_rect_grasp_group(self,
                                 hrois,
                                 cls_score,
                                 score_pred,
                                 bbox_pred,
                                 origin_depth,
                                 img_shape,
                                 ori_shape,
                                 scale_factor,
                                 dataset='graspnet',
                                 center_crop=False,
                                 center_crop_xstart=None,
                                 center_crop_ystart=None,
                                 rescale=False,
                                 cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        #
        # if hrois.size(1) == 6:
        #     hbboxes = xyxydepth_to_xywhthetadepth(hrois[:, 1:])
        # elif hrois.size(1) == 7:
        #     hbboxes = hrois[:, 1:]
        # else:
        #     assert hrois.size(1) != 6 and hrois.size(1) != 7, "hrois.size(1) must be 6 or 7"

        if hrois.size(1) == 5:
            hbboxes = hbbox_to_xywhthetaz(hrois[:, 1:], origin_depth, ori_shape)
        elif hrois.size(1) == 6:
            hbboxes = hrois[:, 1:]
        else:
            assert hrois.size(1) != 5 and hrois.size(1) != 6, "hrois.size(1) must be 5 or 6"

        rbboxes, mask_depth = self.bbox_coder.decode(hbboxes, bbox_pred)

        if rescale and rbboxes.size(0) > 0:
            scale_factor = rbboxes.new_tensor(scale_factor)
            rbboxes[:, :4] /= scale_factor

        if center_crop:
            rbboxes[:, 0] += center_crop_xstart
            rbboxes[:, 1] += center_crop_ystart
        # graspnet
        if dataset == 'graspnet':
            # mask = scores[:, 0] > cfg.score_threshold
            # rbboxes_poly = xywhtheta_to_points_graspnet(rbboxes, ori_shape)
            # rect_grasps, _ = multiclass_poly_nms_8_points_graspnet(rbboxes_poly, scores, score_pred,
            #                                               cfg.nms,
            #                                               score_thr=cfg.score_threshold,
            #                                               max_num=cfg.max_per_img)
            # rect_grasps = xywhtheta_to_rect_grasp_group(rect_grasps[:, :8].detach().cpu().numpy(),
            #                                             rect_grasps[:, 8:9].detach().cpu().numpy(), ori_shape)
            # rect_grasps_xywhtheta = points_to_xywhtheta_graspnet(rect_grasps[:, :8])
            mask = scores[:, 0][mask_depth] > cfg.score_threshold
            rbboxes = rbboxes[mask].detach().cpu().numpy()
            scores = scores[mask_depth][mask].detach().cpu().numpy()
            score_pred = score_pred[mask_depth][mask].detach().cpu().numpy()
            origin_depth = origin_depth.detach().cpu().numpy()
            rect_grasps, _ = xywhthetadepth_to_rect_grasp_group(rbboxes, score_pred, ori_shape, depth_map=origin_depth)
            return rect_grasps #, rbboxes, scores
        else:
            rbboxes_poly = xywhtheta_to_points_graspnet(rbboxes, ori_shape)
            rect_grasps, _ = multiclass_poly_nms_8_points(rbboxes_poly, scores,
                                                          cfg.nms,
                                                          score_thr=cfg.score_threshold,
                                                          max_num=cfg.max_per_img)
            rect_grasps_xywhtheta = points_to_xywhtheta_graspnet(rect_grasps[:, :8])
            rect_scores = rect_grasps[:, 8].reshape(-1, 1)
            rect_grasps_xywhtheta = torch.cat((rect_grasps_xywhtheta, rect_scores), dim=1).detach().cpu().numpy()
            return rect_grasps_xywhtheta




