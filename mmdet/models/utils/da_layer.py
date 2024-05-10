# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from ..builder import build_loss
from .gradient_reverse_layer import GradientReverseLayer

class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        t = F.relu(self.conv1_da(x))
        img_features = self.conv2_da(t)
        return img_features


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self,
                 img_head_in_channels,
                 ins_head_in_channels,
                 roi_feat_size=7,
                 da_img_grl_weight=1.0,
                 da_ins_grl_weight=1.0,
                 loss_da_img=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_da_ins=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_da_cst=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                 ):
        super(DomainAdaptationModule, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=roi_feat_size, stride=roi_feat_size)

        self.grl_img = GradientReverseLayer(-1.0 * da_img_grl_weight)
        self.grl_ins = GradientReverseLayer(-1.0 * da_ins_grl_weight)
        self.grl_img_consist = GradientReverseLayer(1.0 * da_img_grl_weight)
        self.grl_ins_consist = GradientReverseLayer(1.0 * da_ins_grl_weight)


        self.imghead = DAImgHead(img_head_in_channels)
        self.inshead = DAInsHead(ins_head_in_channels)
        self.img_da_loss = build_loss(loss_da_img)
        self.ins_da_loss = build_loss(loss_da_ins)
        self.cst_da_loss = build_loss(loss_da_cst)


    def forward_train(self, img_features, da_ins_features, da_ins_labels, img_metas, ins_nums):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)
        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        da_ins_features = self.avgpool(da_ins_features)
        da_ins_features = da_ins_features.view(da_ins_features.size(0), -1)

        img_grl_fea = self.grl_img(img_features)
        ins_grl_fea = self.grl_ins(da_ins_features)
        img_grl_consist_fea = self.grl_img_consist(img_features)
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_features)

        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)
        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_img_consist_features = da_img_consist_features.sigmoid()
        da_ins_consist_features = da_ins_consist_features.sigmoid()

        num_imgs = da_img_features.shape[0]
        da_img_labels = []
        da_img_targets = []
        da_img_weights = []
        for i in range(num_imgs):
            da_img_target = da_img_features[i].permute(1, 2, 0).reshape(-1)
            if img_metas[i]['domain'] == 'source':
                da_img_label = da_img_target.new_ones((da_img_target.shape[-1], ),
                                                      dtype=torch.long)
            else:
                da_img_label = da_img_target.new_zeros((da_img_target.shape[-1], ),
                                                       dtype=torch.long)
            da_img_weight = da_img_target.new_ones(da_img_target.shape[-1], dtype=torch.float)
            da_img_targets.append(da_img_target)
            da_img_labels.append(da_img_label)
            da_img_weights.append(da_img_weight)

        da_img_targets = torch.cat(da_img_targets, 0)
        da_img_labels = torch.cat(da_img_labels, 0)
        da_img_weights = torch.cat(da_img_weights, 0)
        da_img_loss = self.img_da_loss(da_img_targets, da_img_labels, da_img_weights, avg_factor=da_img_weights.size(0))

        da_ins_weights = da_ins_features.new_ones(len(da_ins_features), dtype=torch.float)
        da_ins_features = da_ins_features.reshape(-1)
        da_ins_loss = self.ins_da_loss(da_ins_features, da_ins_labels, da_ins_weights, avg_factor=da_ins_weights.size(0))

        # len_ins = len(da_ins_labels)
        # intervals = [torch.nonzero(da_ins_labels).size(0), len_ins-torch.nonzero(da_ins_labels).size(0)]
        da_img_consist_means = []
        for i in range(num_imgs):
            da_img_consist_mean = torch.mean(da_img_consist_features[i].reshape(-1)).repeat(ins_nums[i])
            da_img_consist_means.append(da_img_consist_mean)
        da_img_consist_means = torch.cat(da_img_consist_means, dim=0)
        da_consist_weights = da_img_consist_means.new_ones(len(da_img_consist_means), dtype=torch.float)
        da_ins_consist_features = da_ins_consist_features.reshape(-1)
        da_cst_loss = self.cst_da_loss(da_img_consist_means,
                                       da_ins_consist_features,
                                       da_consist_weights,
                                       avg_factor=da_img_consist_means.size(0))

        losses = dict(loss_img_da=da_img_loss,
                      loss_ins_da=da_ins_loss,
                      loss_cst_da=da_cst_loss)
        return losses




class RGBDDomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self,
                 rgb_head_in_channels,
                 depth_head_in_channels,
                 ins_head_in_channels,
                 roi_feat_size=7,
                 da_rgb_grl_weight=1.0,
                 da_depth_grl_weight=1.0,
                 da_ins_grl_weight=1.0,
                 loss_da_rgb=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_da_depth=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_da_ins=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_da_rgb_cst=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_da_depth_cst=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                 ):
        super(RGBDDomainAdaptationModule, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=roi_feat_size, stride=roi_feat_size)

        self.grl_rgb = GradientReverseLayer(-1.0 * da_rgb_grl_weight)
        self.grl_depth = GradientReverseLayer(-1.0 * da_depth_grl_weight)
        self.grl_ins = GradientReverseLayer(-1.0 * da_ins_grl_weight)

        self.grl_rgb_consist = GradientReverseLayer(1.0 * da_rgb_grl_weight)
        self.grl_depth_consist = GradientReverseLayer(1.0 * da_depth_grl_weight)
        self.grl_ins_consist = GradientReverseLayer(1.0 * da_ins_grl_weight)

        self.rgbhead = DAImgHead(rgb_head_in_channels)
        self.depthhead = DAImgHead(depth_head_in_channels)
        self.inshead = DAInsHead(ins_head_in_channels)
        self.rgb_da_loss = build_loss(loss_da_rgb)
        self.depth_da_loss = build_loss(loss_da_depth)
        self.ins_da_loss = build_loss(loss_da_ins)
        self.rgb_cst_da_loss = build_loss(loss_da_rgb_cst)
        self.depth_cst_da_loss = build_loss(loss_da_depth_cst)


    def forward_train(self, rgb_features, depth_features, da_ins_features, da_ins_labels, img_metas, ins_nums):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)
        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        da_ins_features = self.avgpool(da_ins_features)
        da_ins_features = da_ins_features.view(da_ins_features.size(0), -1)

        rgb_grl_fea = self.grl_rgb(rgb_features)
        depth_grl_fea = self.grl_depth(depth_features)
        ins_grl_fea = self.grl_ins(da_ins_features)
        rgb_grl_consist_fea = self.grl_rgb_consist(rgb_features)
        depth_grl_consist_fea = self.grl_depth_consist(depth_features)
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_features)

        da_rgb_features = self.rgbhead(rgb_grl_fea)
        da_depth_features = self.depthhead(depth_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)
        da_rgb_consist_features = self.rgbhead(rgb_grl_consist_fea)
        da_depth_consist_features = self.depthhead(depth_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_rgb_consist_features = da_rgb_consist_features.sigmoid()
        da_depth_consist_features = da_depth_consist_features.sigmoid()
        da_ins_consist_features = da_ins_consist_features.sigmoid()


        num_imgs = da_rgb_features.shape[0]
        da_rgb_labels = []
        da_rgb_targets = []
        da_rgb_weights = []

        # front_weights = []
        # back_weights = []

        for i in range(num_imgs):

            # front_flags = da_rgb_features.new_ones((da_rgb_features[i].shape[1], da_rgb_features[i].shape[2]),
            #                                        dtype=torch.long)
            # for tmpi in range(front_flags.shape[0]):
            #     for tmpj in range(front_flags.shape[1]):
            #         if segment[i][0][16 * tmpi + 8][16 * tmpj + 8] == 0:
            #             front_flags[tmpi][tmpj] = 0
            # front_weight = front_flags.reshape(-1)
            # front_weights.append(front_weight)
            # back_weight = 1 - front_weight
            # back_weights.append(back_weight)

            da_rgb_target = da_rgb_features[i].permute(1, 2, 0).reshape(-1)
            if img_metas[i]['domain'] == 'source':
                da_rgb_label = da_rgb_target.new_ones((da_rgb_target.shape[-1], ),
                                                      dtype=torch.long)
            else:
                da_rgb_label = da_rgb_target.new_zeros((da_rgb_target.shape[-1], ),
                                                       dtype=torch.long)
            da_rgb_weight = da_rgb_target.new_ones(da_rgb_target.shape[-1], dtype=torch.float)
            da_rgb_targets.append(da_rgb_target)
            da_rgb_labels.append(da_rgb_label)
            da_rgb_weights.append(da_rgb_weight)
        da_rgb_targets = torch.cat(da_rgb_targets, 0)
        da_rgb_labels = torch.cat(da_rgb_labels, 0)
        da_rgb_weights = torch.cat(da_rgb_weights, 0)
        da_rgb_loss = self.rgb_da_loss(da_rgb_targets, da_rgb_labels, da_rgb_weights, avg_factor=da_rgb_weights.size(0))

        da_depth_labels = []
        da_depth_targets = []
        da_depth_weights = []
        for i in range(num_imgs):
            da_depth_target = da_depth_features[i].permute(1, 2, 0).reshape(-1)
            if img_metas[i]['domain'] == 'source':
                da_depth_label = da_depth_target.new_ones((da_depth_target.shape[-1],),
                                                      dtype=torch.long)
            else:
                da_depth_label = da_depth_target.new_zeros((da_depth_target.shape[-1],),
                                                       dtype=torch.long)
            da_depth_weight = da_depth_target.new_ones(da_depth_target.shape[-1], dtype=torch.float)
            da_depth_targets.append(da_depth_target)
            da_depth_labels.append(da_depth_label)
            da_depth_weights.append(da_depth_weight)
        da_depth_targets = torch.cat(da_depth_targets, 0)
        da_depth_labels = torch.cat(da_depth_labels, 0)
        da_depth_weights = torch.cat(da_depth_weights, 0)
        da_depth_loss = self.depth_da_loss(da_depth_targets, da_depth_labels, da_depth_weights, avg_factor=da_depth_weights.size(0))

        da_ins_weights = da_ins_features.new_ones(len(da_ins_features), dtype=torch.float)
        da_ins_features = da_ins_features.reshape(-1)
        da_ins_loss = self.ins_da_loss(da_ins_features, da_ins_labels, da_ins_weights, avg_factor=da_ins_weights.size(0))


        da_rgb_consist_means = []
        da_depth_consist_means = []
        for i in range(num_imgs):
            da_rgb_consist_mean = torch.mean(da_rgb_consist_features[i].reshape(-1)).repeat(ins_nums[i])
            da_depth_consist_mean = torch.mean(da_depth_consist_features[i].reshape(-1)).repeat(ins_nums[i])
            da_rgb_consist_means.append(da_rgb_consist_mean)
            da_depth_consist_means.append(da_depth_consist_mean)
        da_rgb_consist_means = torch.cat(da_rgb_consist_means, dim=0)
        da_depth_consist_means = torch.cat(da_depth_consist_means, dim=0)
        da_rgb_consist_weights = da_rgb_consist_means.new_ones(len(da_rgb_consist_means), dtype=torch.float)
        da_depth_consist_weights = da_depth_consist_means.new_ones(len(da_depth_consist_means), dtype=torch.float)
        da_ins_consist_features = da_ins_consist_features.reshape(-1)
        da_rgb_cst_loss = self.rgb_cst_da_loss(da_rgb_consist_means,
                                       da_ins_consist_features,
                                       da_rgb_consist_weights,
                                       avg_factor=da_rgb_consist_means.size(0))
        da_depth_cst_loss = self.depth_cst_da_loss(da_depth_consist_means,
                                               da_ins_consist_features,
                                               da_depth_consist_weights,
                                               avg_factor=da_depth_consist_means.size(0))

        losses = dict(loss_rgb_da=da_rgb_loss,
                      loss_depth_da=da_depth_loss,
                      loss_ins_da=da_ins_loss,
                      loss_cst_rgb_da=da_rgb_cst_loss,
                      loss_cst_depth_da=da_depth_cst_loss)

        # losses = dict(loss_ins_da=da_ins_loss)


        # front_weights = torch.cat(front_weights)
        # back_weights = torch.cat(back_weights)
        # losses['loss_rgb_front_da'] = self.rgb_da_loss(da_rgb_targets, da_rgb_labels, front_weights,
        #                                                avg_factor=torch.sum(front_weights)).detach()
        # losses['loss_rgb_back_da'] = self.rgb_da_loss(da_rgb_targets, da_rgb_labels, back_weights,
        #                                                avg_factor=torch.sum(back_weights)).detach()
        # losses['loss_depth_front_da'] = self.rgb_da_loss(da_depth_targets, da_depth_labels, front_weights,
        #                                                avg_factor=torch.sum(front_weights)).detach()
        # losses['loss_depth_back_da'] = self.rgb_da_loss(da_depth_targets, da_depth_labels, back_weights,
        #                                               avg_factor=torch.sum(back_weights)).detach()


        return losses
