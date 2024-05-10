# --------------------------------------------------------
# Copyright (c) 2018 Xi'an Jiaotong University
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Variable

DEBUG = False

class RoisPairExpandLayer(nn.Module):
    def __init__(self):
        super(RoisPairExpandLayer, self).__init__()

    def forward(self, rois, batch_size, object_num):
        """
        :param rois: region of intrests list
        :param batch_size: image number in one batch
        :param obj_num: a Tensor that indicates object numbers in each image
        :return:
        """
        self.rois = torch.Tensor([]).type_as(rois).float()
        for img_num in range(object_num.size(0)):
            begin_idx = object_num[:img_num].sum().item()
            if object_num[img_num] == 1:
                cur_rois = rois[int(begin_idx):int(begin_idx + object_num[img_num].item())][:, 1:5]
                cur_rois = torch.cat([((img_num % batch_size) * torch.ones(cur_rois.size(0), 1)).type_as(cur_rois),
                                      cur_rois], 1)
                self.rois = torch.cat([self.rois, cur_rois], 0)
            elif object_num[img_num] > 1:
                cur_rois = rois[int(begin_idx):int(begin_idx + object_num[img_num].item())][:, 1:5]
                cur_rois = self._single_image_expand(cur_rois)
                cur_rois = torch.cat([((img_num % batch_size) * torch.ones(cur_rois.size(0), 1)).type_as(cur_rois),
                                      cur_rois], 1)
                self.rois = torch.cat([self.rois, cur_rois], 0)

        return self.rois

    def backward(self):
        pass

    def _single_image_expand(self, rois):
        _rois = rois
        _rois_num = _rois.size(0)
        for b1 in range(_rois_num):
            for b2 in range(b1+1, _rois_num):
                if b1 != b2:
                    box1 = rois[b1]
                    box2 = rois[b2]
                    tmax = torch.max(box1[2:4], box2[2:4])
                    tmin = torch.min(box1[0:2], box2[0:2])
                    unionbox = torch.cat([tmin, tmax],0)
                    unionbox = torch.reshape(unionbox, (-1, 4))
                    _rois = torch.cat([_rois, unionbox], 0)
        return _rois
