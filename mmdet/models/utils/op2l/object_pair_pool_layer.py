# --------------------------------------------------------
# Copyright (c) 2018 Xi'an Jiaotong University
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

from __future__ import absolute_import
import random
import torch
import torch.nn as nn
from mmcv import ops

from .rois_pair_expand_layer import RoisPairExpandLayer
from .object_pair_layer import ObjectPairLayer
from ..builder import OBJECTPAIRPOOLLAYER
from ...builder import build_roi_extractor

@OBJECTPAIRPOOLLAYER.register_module()
class ObjectPairPoolLayer(nn.Module):
    def __init__(self,
                 roi_extractor,
                 isex):
        super(ObjectPairPoolLayer, self).__init__()
        self.isex = isex

        self.op2l_rois_pairing = RoisPairExpandLayer()
        self.op2l_object_pair = ObjectPairLayer(self.isex)

        self.roi_extractor = build_roi_extractor(roi_extractor)

    def init_weights(self):
        pass

    def forward(self, feats, rois, batch_size, object_num):
        """
        :param feats: input features from basenet Channels: x W x H
        :param rois: object detection results: N x 4
        :return: object pair features: N(N-1) x 3 x 512 x 7 x 7

        algorithm:
        1 regions of intrests pairing
        2 roi pooling
        3 concate object pair features
        """
        paired_rois = self.op2l_rois_pairing(rois, batch_size, object_num)
        pooled_feat = self.roi_extractor(
            feats[:self.roi_extractor.num_inputs], paired_rois)
        obj_pair_feats = self.op2l_object_pair(pooled_feat, batch_size, object_num)
        return obj_pair_feats, paired_rois