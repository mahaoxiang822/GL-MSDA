from mmdet.models.builder import HEADS
import torch
import torch.nn as nn
from mmcv.cnn import (normal_init, kaiming_init, constant_init)
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from mmdet.core import build_anchor_generator, build_bbox_coder
from mmdet.models.builder import build_loss
from mmdet.core.bbox.grasp_transforms import points_to_xywhtheta, xywhtheta_to_points
from mmdet.models.backbones.resnet import Bottleneck
import pdb
import numpy as np
from torch.autograd import Variable

@HEADS.register_module()
class GraspHead(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 bbox_dim,
                 oriented_anchor_generator=dict(
                     type='OrientedAnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     angles=[1]),
                 grasp_coder=dict(
                     type='DeltaXYWHThetaBBoxCoder',
                     target_means=(0, 0, 0, 0, 0),
                     target_stds=(1, 1, 1, 1, 1),
                     angle_factor=180),
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 style='pytorch'):
        super(GraspHead, self).__init__()
        self.convs = nn.Sequential(
            Bottleneck(in_channels, int(in_channels / 4), style=style),
            Bottleneck(in_channels, int(in_channels / 4), style=style),
            Bottleneck(in_channels, int(in_channels / 4), style=style)
        )
        self.oriented_anchor_generator = build_anchor_generator(oriented_anchor_generator)
        self.grasp_coder = build_bbox_coder(grasp_coder)
        self.num_anchors = len(oriented_anchor_generator['scales']) * len(oriented_anchor_generator['ratios']) * len(oriented_anchor_generator['angles'])
        self.num_classes = num_classes
        self.bbox_dim = bbox_dim
        self.conv_reg = nn.Conv2d(in_channels, self.num_anchors * self.bbox_dim, kernel_size=3, padding=1)
        self.conv_cls = nn.Conv2d(in_channels, self.num_anchors * self.num_classes, kernel_size=3, padding=1)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)


    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.001)
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
        for m in self.convs.modules():
            if isinstance(m, Bottleneck):
                constant_init(m.norm3, 0)

    def forward(self, x):
        x = self.convs(x)
        cls_score = self.conv_cls(x).permute(0, 2, 3, 1)
        bbox_pred = self.conv_reg(x).permute(0, 2, 3, 1)
        return cls_score, bbox_pred

    def assign_rois_grasps(self, grasp, grasp_inds, rois_inds):
        """
                :param grasp: list(N_{Gr_gt} x Gdim)
                :param grasp_inds: list(N_{Gr_gt})
                :param rois_inds: bs x N_{rois}
                :return: grasp: bs x N_{rois} x N_{Gr_gt} x Gdim
                """
        # 得到每一个roi对应的ground truth grasp, roi object ind == gt grasp ind
        # bs x N x N_{Gr_gt} x 1
        grasp_mask = (grasp_inds.unsqueeze(-2) == rois_inds.unsqueeze(-1)).unsqueeze(3).float()
        # bs x 1 x N_{Gr_gt} x 5
        grasp = grasp.unsqueeze(1)
        # bs*N x N_{Gr_gt} x 5
        grasp_out = (grasp_mask * grasp).contiguous(). \
            view(rois_inds.size(0) * rois_inds.size(1), grasp_inds.size(1), -1)
        return grasp_out

    def match_gt_anchor(self, anchors, gt_grasps, xthresh, ythresh, angle_thresh, eps=1e-14):
        """
        :param priors: bs x K x 5
        :param gt: bs x N x 5
        :param angle_thresh:
        :return:
        """

        num_anchors = anchors.size(1)
        angle_thresh = angle_thresh / 180 * np.pi

        x_gt = gt_grasps[:, :, 0:1].transpose(2,1)
        y_gt = gt_grasps[:, :, 1:2].transpose(2,1)
        ang_gt = gt_grasps[:, :, 4:5].transpose(2,1)
        mask_gt = (torch.sum(gt_grasps == 0, 2, keepdim=True) != gt_grasps.size(2)).transpose(2, 1)

        xdiff = torch.abs(anchors[:, :, 0:1] - x_gt)
        ydiff = torch.abs(anchors[:, :, 1:2] - y_gt)
        angdiff = torch.abs(anchors[:, :, 4:5] - ang_gt)

        mask = torch.zeros_like(xdiff) + mask_gt.float()

        match_mat = (xdiff <= xthresh) \
                    & (ydiff <= ythresh) \
                    & (angdiff <= angle_thresh) \
                    & (mask != 0)


        match_num = torch.sum(match_mat, 2, keepdim = True)
        label = torch.zeros(self.batch_size, num_anchors).type_as(gt_grasps).long()
        label[(torch.sum(match_mat, 2) > 0)] = 1

        # bs x N x K ->  K x bs x N ->  K x bs x N x 1
        match_mat = match_mat.permute(2, 0, 1).unsqueeze(3)
        # bs x K x 5 ->  K x bs x 5 ->  K x bs x 1 x 5
        gt_grasps = gt_grasps.permute(1, 0, 2).unsqueeze(2)
        # K x bs x N x 5 -> bs x N x 5
        # When a prior matches multi gts, it will use
        # the mean of all matched gts as its target.
        loc = torch.sum(match_mat.float() * gt_grasps, dim=0) + eps

        # make all nans zeros
        keep = (match_num > 0).squeeze()
        loc[keep] /= match_num[keep].float()
        loc = self.grasp_coder.encode(anchors, loc)

        return loc, label

    def mine_hard_samples(self, cls_score_label, cls_score, train_cfg):
        """
        :param loc_t: bs x N x 5
        :param conf_t: bs x N
        :param conf: bs x N x 2
        :return:
        """
        pos = (cls_score_label > 0)
        batch_cls_score = cls_score.data.view(-1, 2)
        loss_c = self._log_sum_exp(batch_cls_score) - batch_cls_score.gather(1, cls_score_label.view(-1, 1))
        loss_c = loss_c.view(self.batch_size, -1)

        loss_c[pos] = -1  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        # To find element indexes that indicate elements which have highest confidence loss
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = train_cfg.neg_pos_ratio * num_pos
        neg = (idx_rank < num_neg.expand_as(idx_rank)) & (pos != 1)

        cls_score_label[neg.eq(0) & pos.eq(0)] = -1

        iw = pos.gt(0).float() * train_cfg.bbox_inside_weight
        iw = iw.unsqueeze(2).expand(cls_score.size(0), -1, 5)

        if train_cfg.bbox_pos_weight < 0:
            ow = (pos + neg).gt(0).float() / ((num_pos + num_neg)|1).float()
            ow = ow.unsqueeze(2).expand(cls_score.size(0), -1, 5)
        else:
            ow = (pos.gt(0).float() * train_cfg.bbox_pos_weight \
                + neg.gt(0).float()) / ((num_pos + num_neg)|1).float()
            ow = ow.unsqueeze(2).expand(cls_score.size(0), -1, 5)

        if torch.isnan(ow.data).sum() > 0:
            pdb.set_trace()

        if (neg.gt(0) & pos.gt(0)).sum().item() > 0:
            pdb.set_trace()

        return iw, ow

    def get_targets(self,
                    cls_score,
                    gt_grasps,
                    anchors,
                    x_thresh,
                    y_thresh,
                    train_cfg):
        self.batch_size = gt_grasps.size(0)

        bbox_target, label = self.match_gt_anchor(anchors, gt_grasps, x_thresh, y_thresh,
                                             train_cfg.angle_thresh,
                                             train_cfg.eps)

        bbox_inside_weights, bbox_outside_weights = self.mine_hard_samples(label, cls_score, train_cfg)

        return bbox_target, label, bbox_inside_weights, bbox_outside_weights

    def _log_sum_exp(self, x):
        """Utility function for computing log_sum_exp while determining
        This will be used to determine unaveraged confidence loss across
        all examples in a batch.
        Args:
            x (Variable(tensor)): conf_preds from conf layers
        """
        x_max, _ = x.data.max(dim = 1, keepdim = True)
        return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max

    def loss(self,
             cls_score,
             bbox_pred,
             grasp_rois_mask,
             grasp_rois,
             rois_inds,
             gt_grasps,
             gt_grasp_inds,
             train_cfg):
        feat_height = cls_score.size(1)
        feat_width = cls_score.size(2)
        grasp_all_anchors = self.oriented_anchor_generator.generate_anchors(
            feat_height, feat_width, grasp_rois)
        grasp_all_anchors = grasp_all_anchors.type_as(gt_grasps[0])
        grasp_all_anchors = grasp_all_anchors[grasp_rois_mask > 0]
        # bs*N x 1 x 1
        rois_w = (grasp_rois[:, :, 3] - grasp_rois[:, :, 1]).data.view(-1).unsqueeze(1).unsqueeze(2)
        rois_h = (grasp_rois[:, :, 4] - grasp_rois[:, :, 2]).data.view(-1).unsqueeze(1).unsqueeze(2)
        rois_w = rois_w[grasp_rois_mask > 0]
        rois_h = rois_h[grasp_rois_mask > 0]
        # bs*N x 1 x 1
        fsx = rois_w / cls_score.size(1)
        fsy = rois_h / cls_score.size(2)
        # bs*N x 1 x 1
        xleft = grasp_rois[:, :, 1].data.view(-1).unsqueeze(1).unsqueeze(2)
        ytop = grasp_rois[:, :, 2].data.view(-1).unsqueeze(1).unsqueeze(2)
        xleft = xleft[grasp_rois_mask > 0]
        ytop = ytop[grasp_rois_mask > 0]

        bbox_pred = bbox_pred.contiguous().view(bbox_pred.size(0), -1, 5)
        cls_score = cls_score.contiguous().view(cls_score.size(0), -1, 2)


        # inside weights indicate which bounding box should be regressed
        # outside weidhts indicate two things:
        # 1. Which bounding box should contribute for classification loss,
        # 2. Balance cls loss and bbox loss
        grasp_gt_xywhc = points_to_xywhtheta(gt_grasps)
        # bs*N x N_{Gr_gt} x 5
        grasp_gt_xywhc = self.assign_rois_grasps(grasp_gt_xywhc, gt_grasp_inds, rois_inds)
        # filter out negative samples
        grasp_gt_xywhc = grasp_gt_xywhc[grasp_rois_mask > 0]

        # absolute coords to relative coords
        grasp_gt_xywhc[:, :, 0:1] -= xleft
        grasp_gt_xywhc[:, :, 0:1] = torch.clamp(grasp_gt_xywhc[:, :, 0:1], min=0)
        grasp_gt_xywhc[:, :, 0:1] = torch.min(grasp_gt_xywhc[:, :, 0:1], rois_w)
        grasp_gt_xywhc[:, :, 1:2] -= ytop
        grasp_gt_xywhc[:, :, 1:2] = torch.clamp(grasp_gt_xywhc[:, :, 1:2], min=0)
        grasp_gt_xywhc[:, :, 1:2] = torch.min(grasp_gt_xywhc[:, :, 1:2], rois_h)

        # grasp training data
        grasp_bbox_target, grasp_label, grasp_bbox_inside_weight, grasp_bbox_outside_weight = self.get_targets(cls_score,
                                                                                grasp_gt_xywhc,
                                                                                grasp_all_anchors,
                                                                                x_thresh=fsx / 2,
                                                                                y_thresh=fsy / 2,
                                                                                train_cfg=train_cfg)
        # 去除正负样本之外的样本，只保留正负样本
        # 分类只采用正负样本
        grasp_keep = Variable(grasp_label.view(-1).ne(-1).nonzero(as_tuple=False).view(-1))
        cls_score = torch.index_select(cls_score.view(-1, 2), 0, grasp_keep.data)
        grasp_label = torch.index_select(grasp_label.view(-1), 0, grasp_keep.data)
        # 定位只采用正样本
        grasp_bbox_target = grasp_bbox_target.view(-1, 5)
        grasp_bbox_weight = grasp_bbox_inside_weight.view(-1, 5)
        bbox_pred = bbox_pred.view(-1, 5)

        num_samples = len(cls_score)
        grasp_cls_loss = self.loss_cls(cls_score, grasp_label)
        grasp_bbox_loss = self.loss_bbox(bbox_pred, grasp_bbox_target,
                                         grasp_bbox_weight, avg_factor=num_samples)
        # grasp_bbox_loss = self.smooth_l1_loss(bbox_pred, grasp_bbox_target, grasp_bbox_inside_weight,
        #                                       grasp_bbox_outside_weight, dim=[2, 1])

        if torch.isnan(grasp_cls_loss) or torch.isnan(grasp_bbox_loss):
            import pdb
            pdb.set_trace()

        return dict(
            grasp_cls_loss=grasp_cls_loss,
            grasp_bbox_loss=grasp_bbox_loss
        )

    def get_grasps(self,
                   cls_score,
                   bbox_pred,
                   grasp_rois,
                   cfg,
                   img_shape,
                   rescale=False,
                   scale_factor=None):
        feat_height = cls_score.size(1)
        feat_width = cls_score.size(2)
        grasp_all_anchors = self.oriented_anchor_generator.generate_anchors(
            feat_height, feat_width, grasp_rois)
        grasp_all_anchors = grasp_all_anchors.type_as(grasp_rois)

        bbox_pred = bbox_pred.contiguous().view(bbox_pred.size(0), -1, 5)
        cls_score = cls_score.contiguous().view(cls_score.size(0), -1, 2)
        cls_prob = F.softmax(cls_score, 2)


        grasp_pred = self.grasp_coder.decode(grasp_all_anchors, bbox_pred)
        # bs*N x K*A x 1
        grasp_rois_w = (grasp_rois[:, :, 3] - grasp_rois[:, :, 1]).data.view(-1). \
            unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 0:1])
        grasp_rois_h = (grasp_rois[:, :, 4] - grasp_rois[:, :, 2]).data.view(-1). \
            unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 1:2])
        keep_mask = (grasp_pred[:, :, 0:1] > 0) & (grasp_pred[:, :, 1:2] > 0) & \
                    (grasp_pred[:, :, 0:1] < grasp_rois_w) & (grasp_pred[:, :, 1:2] < grasp_rois_h)
        # grasp_pred[:, :, 0:1] = torch.clamp(grasp_pred[:, :, 0:1], min=0)
        # grasp_pred[:, :, 0:1] = torch.min(grasp_pred[:, :, 0:1], grasp_rois_w)
        # grasp_pred[:, :, 1:2] = torch.clamp(grasp_pred[:, :, 1:2], min=0)
        # grasp_pred[:, :, 1:2] = torch.min(grasp_pred[:, :, 1:2], grasp_rois_h)
        grasp_scores = (cls_prob).contiguous(). \
            view(grasp_rois.size(0), grasp_rois.size(1), -1, 2)
        # bs*N x 1 x 1
        xleft = grasp_rois[:, :, 1].data.view(-1).unsqueeze(1).unsqueeze(2)
        ytop = grasp_rois[:, :, 2].data.view(-1).unsqueeze(1).unsqueeze(2)
        # rois offset
        grasp_pred[:, :, 0:1] = grasp_pred[:, :, 0:1] + xleft
        grasp_pred[:, :, 1:2] = grasp_pred[:, :, 1:2] + ytop
        # bs x N x K*A x 8
        grasp_pred_bboxes = xywhtheta_to_points(grasp_pred, max_shape=None).contiguous().view(grasp_rois.size(0), grasp_rois.size(1), -1, 8)
        # bs x N x K*A
        grasp_pos_scores = grasp_scores[:, :, :, 1]
        # bs x N x K*A
        _, grasp_score_idx = torch.sort(grasp_pos_scores, dim=2, descending=True)
        _, grasp_idx_rank = torch.sort(grasp_score_idx)
        # bs x N x K*A mask
        grasp_max_score_mask = (grasp_idx_rank < cfg.topn)
        # bs x N x topN
        grasp_max_scores = grasp_scores[:, :, :, 1][grasp_max_score_mask].contiguous(). \
            view(grasp_rois.size()[:2] + (cfg.topn,))
        # scores = scores * grasp_maxscores[:, :, 0:1]
        # bs x N x topN x 8
        grasp_pred_bboxes = grasp_pred_bboxes[grasp_max_score_mask].view(grasp_rois.size()[:2] + (cfg.topn, 8))
        grasp_pred_bboxes = grasp_pred_bboxes.squeeze()
        grasp_max_scores = grasp_max_scores.squeeze()
        if rescale and scale_factor is not None:
            scale_factor = grasp_pred_bboxes.new_tensor(scale_factor)
            grasp_pred_bboxes = (grasp_pred_bboxes.view(grasp_pred_bboxes.size(0), -1, 8)/
                                scale_factor).view(grasp_pred_bboxes.size(0), -1)
        return grasp_pred_bboxes, grasp_max_scores



    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
        in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = out_loss_box
        for i in sorted(dim, reverse=True):
            loss_box = loss_box.sum(i)
        loss_box = loss_box.mean()
        return loss_box