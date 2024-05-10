import torch.nn as nn
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import (constant_init, kaiming_init, uniform_init)
import os
import cv2
import torchvision
import numpy as np
import torch.nn.functional as F
import math
from sklearn.decomposition import PCA


class SpatialAttentionLayer(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False):
        super(SpatialAttentionLayer, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x, mask=None, img_metas=None):
        if mask is not None:
            h, w = x.shape[-2], x.shape[-1]
            mask = F.interpolate(mask, (h, w), mode='bilinear')
            x_out = x * mask
            return x_out
        x_out_max = torch.max(x, dim=1)[0].unsqueeze(1)
        x_out_mean = torch.mean(x, dim=1).unsqueeze(1)
        x_out = torch.cat((x_out_max, x_out_mean), dim=1)
        x_out = self.conv(x_out)
        if self.bn is not None:
            x_out = self.bn(x_out)
        if self.relu is not None:
            x_out = self.relu(x_out)
        x_out = torch.sigmoid(x_out)

        if img_metas is not None:
            dir = os.path.join('/home/qinran_2020/mmdetection_grasp/eval/spatial_attention_mask/')
            scene_dir = os.path.join(dir, 'scene_%04d' % img_metas[0]['sceneId'])
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            img_mask_name = os.path.join(scene_dir, '%04d.png' % img_metas[0]['annId'])
            img_mask = x_out[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            heatmap_mask = None
            heatmap_mask = cv2.normalize(img_mask, heatmap_mask, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_mask = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)
            cv2.imwrite(img_mask_name, heatmap_mask)

        x_out = x * x_out
        return x_out


class GateAttentionLayer(nn.Module):

    def __init__(self,
                 in_planes):
        super(GateAttentionLayer, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes * 2,
        #                        in_planes * 2 // 16,
        #                        kernel_size=(1, 1),
        #                        stride=1,
        #                        padding=0,
        #                        dilation=1,
        #                        groups=1,
        #                        bias=False)
        # self.conv2 = nn.Conv2d(in_planes * 2 // 16,
        #                        1,
        #                        kernel_size=(3, 3),
        #                        stride=1,
        #                        padding=1,
        #                        dilation=1,
        #                        groups=1,
        #                        bias=False)

        # self.conv1 = nn.Conv2d(in_planes * 2,
        #                        in_planes,
        #                        kernel_size=(1, 1),
        #                        stride=1,
        #                        padding=0,
        #                        dilation=1,
        #                        groups=1,
        #                        bias=False)
        # self.conv2 = nn.Conv2d(2,
        #                        1,
        #                        kernel_size=7,
        #                        stride=1,
        #                        padding=3,
        #                        dilation=1,
        #                        groups=1,
        #                        bias=False)

        self.conv1 = nn.Conv2d(in_planes * 2,
                               in_planes,
                               kernel_size=(1, 1),
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1,
                               dilation=1,
                               groups=1,
                               bias=False)

        self.relu = nn.ReLU(inplace=True)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x1, x2):
        # x = torch.cat((x1, x2), dim=1)
        # x = self.conv1(x)
        # x = self.relu(x)
        # x_max = torch.max(x, dim=1)[0].unsqueeze(1)
        # x_mean = torch.mean(x, dim=1).unsqueeze(1)
        # x = torch.cat((x_max, x_mean), dim=1)
        # x = self.conv2(x)
        # x = torch.sigmoid(x)
        # x = x * x2
        # return x

        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = x * x2
        return x

# class CrossModalMultiHeadAttention_origin(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  num_head,
#                  ratio):
#         super(CrossModalMultiHeadAttention_origin, self).__init__()
#         self.in_channels = in_channels
#         self.num_head = num_head
#         self.out_channels = int(num_head * in_channels * ratio)
#         self.query_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
#         self.key_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
#         self.value_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
#         self.W = nn.Sequential(
#             nn.Conv2d(in_channels=int(in_channels * ratio), out_channels=in_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(in_channels)
#         )
#         self.fuse = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1)
#             # nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1, bias=False)
#         )
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 kaiming_init(m)
#             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#                 constant_init(m, 1)
#
#     def forward(self, key, query, img_metas=None):
#         # key:RGB; query:Depth
#         batch, channels, height, width = query.size()
#         q_out = self.query_conv(query).contiguous().view(batch, self.num_head, -1, height, width)
#         k_out = self.key_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
#         v_out = self.value_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
#
#         att = (q_out * k_out).sum(dim=2) / np.sqrt(self.out_channels)
#
#         if self.num_head == 1:
#             softmax = att.unsqueeze(dim=2)
#         else:
#             softmax = F.softmax(att, dim=1).unsqueeze(dim=2)
#
#
#         weighted_value = v_out * softmax
#         weighted_value = weighted_value.sum(dim=1)
#         out = self.W(weighted_value)
#
#         debug = True
#         if debug and img_metas is not None:
#             dir = os.path.join('/home/qinran_2020/mmdetection_grasp/eval/cross_attention_origin/')
#             scene_dir = os.path.join(dir, 'scene_%04d' % img_metas[0]['sceneId'])
#             if not os.path.exists(scene_dir):
#                 os.makedirs(scene_dir)
#             img_name = os.path.join(scene_dir, '%04d' % img_metas[0]['annId'])
#             img = att[0][0].detach().cpu().numpy().astype(np.float32)
#             img[img < -1] = 0
#             img[img > 1] = 0
#             img = np.abs(img)
#             # img = np.mean(img, axis=0)
#             heatmap = None
#             heatmap = cv2.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#             heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#             cv2.imwrite(img_name + "_attention.png", heatmap)
#             # channels = len(key[0])
#             # for i in range(channels):
#             #     img = key[0][i].detach().cpu().numpy().astype(np.float32)
#             #     heatmap = None
#             #     heatmap = cv2.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#             #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#             #     cv2.imwrite(img_name + "_%d.png" % i, heatmap)
#
#         return self.fuse(torch.cat([key, out], dim=1))

class CrossModalMultiHeadAttention_origin(nn.Module):

    def __init__(self,
                 in_channels,
                 num_head,
                 ratio):
        super(CrossModalMultiHeadAttention_origin, self).__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.out_channels = int(num_head * in_channels * ratio)
        self.query_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=False)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels * ratio), out_channels=in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.fuse = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels, in_channels, kernel_size=1)
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1, bias=False)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, key, query, img_metas=None):
        tmp = key
        key = query
        query = tmp
        batch, channels, height, width = query.size()
        q_out = self.query_conv(query).contiguous().view(batch, self.num_head, -1, height, width)
        k_out = self.key_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
        v_out = self.value_conv(key).contiguous().view(batch, self.num_head, -1, height, width)

        att = (q_out * k_out).sum(dim=2) / np.sqrt(self.out_channels)

        if self.num_head == 1:
            softmax = torch.sigmoid(att).unsqueeze(dim=2)
            # softmax = att.unsqueeze(dim=2)
        else:
            softmax = F.softmax(att, dim=1).unsqueeze(dim=2)


        weighted_value = v_out * softmax
        weighted_value = weighted_value.sum(dim=1)
        out = self.W(weighted_value)

        debug = False
        if debug and img_metas is not None:
            dir = os.path.join('/home/qinran_2020/mmdetection_grasp/eval/cross_attention/')
            scene_dir = os.path.join(dir, 'scene_%04d' % img_metas[0]['sceneId'])
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            img_name = os.path.join(scene_dir, '%04d' % img_metas[0]['annId'])
            img = torch.sigmoid(att)[0][0].detach().cpu().numpy().astype(np.float32)
            # img[img < -1] = 0
            # img[img > 1] = 0
            heatmap = None
            heatmap = cv2.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(img_name + "_attention.png", heatmap)
        # return self.fuse(torch.cat([key, out], dim=1))
        return self.fuse(torch.cat([query, out], dim=1))

class CrossModalMultiHeadAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 num_head,
                 ratio):
        super(CrossModalMultiHeadAttention, self).__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.out_channels = int(in_channels * ratio)
        self.query_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
        self.key_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
        self.value_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
        self.W = nn.Conv2d(in_channels=self.out_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.fuse = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels, in_channels, kernel_size=1)
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1, bias=False)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, key, query, img_metas=None):
        # key:RGB; query:Depth
        batch, channels, height, width = query.size()
        q_out = self.query_conv(query).contiguous().view(batch, self.num_head, -1, height, width)
        k_out = self.key_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
        v_out = self.value_conv(query).contiguous().view(batch, self.num_head, -1, height, width)

        att = (q_out * k_out).sum(dim=2) / np.sqrt(self.out_channels // self.num_head)

        if self.num_head == 1:
            softmax = att.unsqueeze(dim=2)
            # softmax = torch.sigmoid(att).unsqueeze(dim=2)
        else:
            # softmax = F.softmax(att, dim=1).unsqueeze(dim=2)
            softmax = torch.sigmoid(att).unsqueeze(dim=2)

        weighted_value = v_out * softmax
        # weighted_value = weighted_value.sum(dim=1)
        weighted_value = weighted_value.view(batch, self.out_channels, height, width)
        out = query + self.W(weighted_value)
        # out = self.W(weighted_value)
        out = self.bn(out)

        debug = False
        if debug and img_metas is not None:
            dir = os.path.join('/home/qinran_2020/mmdetection_grasp/eval/cross_attention_sigmoid/attention')
            scene_dir = os.path.join(dir, 'scene_%04d' % img_metas[0]['sceneId'])
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            img_name = os.path.join(scene_dir, '%04d' % img_metas[0]['annId'])

            img = torch.sigmoid(att)[0][0].detach().cpu().numpy().astype(np.float32)
            # img[img > 0.7] = 0.5
            # img[img < -1] = 0
            # img = np.mean(img, axis=0)
            heatmap = None
            heatmap = cv2.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(img_name + "_attention.png", heatmap)

            # channels = len(out[0])
            # for i in range(channels):
            #     img = out[0][i].detach().cpu().numpy().astype(np.float32)
            #     heatmap = None
            #     heatmap = cv2.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            #     cv2.imwrite(img_name + "_%d.png" % i, heatmap)

        return self.fuse(torch.cat([key, out], dim=1))


class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Default 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Default False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Default 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Default 1e-6.
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 with_proj=False):
        super(SinePositionalEncoding, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.with_proj = with_proj
        if self.with_proj:
            self.proj = nn.Conv2d(num_feats * 2, num_feats * 2, kernel_size=1, bias=True)

    def forward(self, mask, feature_map):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1).type_as(feature_map)
        x_embed = not_mask.cumsum(2).type_as(feature_map)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_feats).type_as(feature_map)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        del x_embed, y_embed, dim_t, pos_x, pos_y
        if self.with_proj:
            pos = self.proj(pos)
        return pos

class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
    """

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.init_weights()

    def init_weights(self):
        """Initialize the learnable weights."""
        uniform_init(self.row_embed)
        uniform_init(self.col_embed)

    def forward(self, mask, feature_map):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

class CrossModalMultiHeadAttentionK(nn.Module):

    def __init__(self,
                 in_channels,
                 num_head,
                 head_dim,
                 sample_nums,
                 with_position_encoding=True,
                 position_encoding_type='sin_abs'):
        super(CrossModalMultiHeadAttentionK, self).__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.sample_nums = sample_nums
        self.head_dim = head_dim
        self.padding = sample_nums // 2
        self.scaling = self.head_dim ** -0.5
        self.proj_channels = head_dim * num_head
        self.with_position_encoding = with_position_encoding
        self.position_encoding_type = position_encoding_type

        if with_position_encoding:
            if position_encoding_type == 'sin_abs':
                self.position_embedding = SinePositionalEncoding(num_feats=self.in_channels // 2,
                                                                 normalize=True)
            if position_encoding_type == 'sin_rel_k':
                self.position_embedding = SinePositionalEncoding(num_feats=self.proj_channels // 2,
                                                                 normalize=True,
                                                                 with_proj=True)
            if position_encoding_type == 'learning_rel_k':
                self.position_embedding = LearnedPositionalEncoding(num_feats=self.proj_channels // 2,
                                                                     row_num_embed=sample_nums,
                                                                     col_num_embed=sample_nums)
        else:
            self.position_embedding = None

        self.query_conv = nn.Conv2d(in_channels, self.proj_channels, kernel_size=1, stride=1, bias=True)
        self.key_conv = nn.Conv2d(in_channels, self.proj_channels, kernel_size=1, stride=1, bias=True)
        self.value_conv = nn.Conv2d(in_channels, self.proj_channels, kernel_size=1, stride=1, bias=True)
        self.out_conv = nn.Conv2d(in_channels=self.proj_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.fuse = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


    def forward(self, key, query, img_metas=None):
        # query:rgb, key:depth
        batch, channels, height, width = query.size()

        padded_key = F.pad(key, [self.padding, self.padding, self.padding, self.padding])
        v_out = self.value_conv(padded_key)

        if self.with_position_encoding:
            if self.position_encoding_type == 'sin_abs':
                mask_query = query.new_zeros((batch, height, width), dtype=torch.bool)
                query_position_embedding = self.position_embedding(mask_query, query)
                query = query + query_position_embedding
                mask_key = key.new_zeros((batch, padded_key.size()[-2], padded_key.size()[-1]), dtype=torch.bool)
                mask_key[:, :self.padding, :] = True
                mask_key[:, :, :self.padding] = True
                mask_key[:, -self.padding:, :] = True
                mask_key[:, :, -self.padding:] = True
                key_position_embedding = self.position_embedding(mask_key, key)
                padded_key = padded_key + key_position_embedding
                del query_position_embedding, mask_query, key_position_embedding, mask_key

        q_out = self.query_conv(query) * self.scaling
        k_out = self.key_conv(padded_key)


        k_out = k_out.unfold(2, self.sample_nums, 1).unfold(3, self.sample_nums, 1)
        v_out = v_out.unfold(2, self.sample_nums, 1).unfold(3, self.sample_nums, 1)

        if self.with_position_encoding:
            if self.position_encoding_type == 'sin_rel_k':
                mask = key.new_zeros((batch, self.sample_nums, self.sample_nums), dtype=torch.bool)
                key_position_embedding = self.position_embedding(mask, key).unsqueeze(-3).unsqueeze(-3)
                k_out = k_out + key_position_embedding
            if self.position_encoding_type == 'learning_rel_k':
                mask = key.new_zeros((batch, self.sample_nums, self.sample_nums), dtype=torch.bool)
                key_position_embedding = self.position_embedding(mask, key).unsqueeze(-3).unsqueeze(-3)
                k_out = k_out + key_position_embedding

        k_out = k_out.contiguous().view(batch, self.num_head, self.head_dim, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.num_head, self.head_dim, height, width, -1)

        q_out = q_out.contiguous().view(batch, self.num_head, self.head_dim, height, width, 1)


        att = F.softmax(q_out*k_out, dim=-1)

        v_out = (v_out * att).sum(dim=-1).view(batch, -1, height, width)

        v_out = self.out_conv(v_out)

        out = self.fuse(torch.cat([query, v_out], dim=1))


        return out, v_out
