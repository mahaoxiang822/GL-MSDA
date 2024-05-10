from ..builder import BACKBONES
import torch.nn as nn
from .resnet import ResNet, BasicBlock, Bottleneck
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.utils import get_root_logger
import torch
from mmdet.models.utils import GateAttentionLayer
import os
import numpy as np
import cv2
from sklearn.decomposition import PCA


@BACKBONES.register_module()
class ResNetFusionSum(ResNet):


    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNetFusionSum, self).__init__(depth=depth,
                                           in_channels=in_channels,
                                           stem_channels=stem_channels,
                                           base_channels=base_channels,
                                           num_stages=num_stages,
                                           strides=strides,
                                           dilations=dilations,
                                           out_indices=out_indices,
                                           style=style,
                                           deep_stem=deep_stem,
                                           avg_down=avg_down,
                                           frozen_stages=frozen_stages,
                                           conv_cfg=conv_cfg,
                                           norm_cfg=norm_cfg,
                                           norm_eval=norm_eval,
                                           dcn=dcn,
                                           stage_with_dcn=stage_with_dcn,
                                           plugins=plugins,
                                           with_cp=with_cp,
                                           zero_init_residual=zero_init_residual)

    def forward(self, x, x1):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            x = x + x1[i]
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class ResNetFusionConcat(ResNet):

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 out_indices_origin=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNetFusionConcat, self).__init__(depth=depth,
                                           in_channels=in_channels,
                                           stem_channels=stem_channels,
                                           base_channels=base_channels,
                                           num_stages=num_stages,
                                           strides=strides,
                                           dilations=dilations,
                                           out_indices=out_indices,
                                           style=style,
                                           deep_stem=deep_stem,
                                           avg_down=avg_down,
                                           frozen_stages=frozen_stages,
                                           conv_cfg=conv_cfg,
                                           norm_cfg=norm_cfg,
                                           norm_eval=norm_eval,
                                           dcn=dcn,
                                           stage_with_dcn=stage_with_dcn,
                                           plugins=plugins,
                                           with_cp=with_cp,
                                           zero_init_residual=zero_init_residual)
        self.out_indices_origin = out_indices_origin
        self.fusion_conv = nn.ModuleList()
        for i in range(num_stages):
            self.fusion_conv.append(nn.Conv2d(
                base_channels * 2 ** (i + 2) * 2,
                base_channels * 2 ** (i + 2),
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False))

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.fusion_conv:
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward_origin(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices_origin:
                outs.append(x)
        return tuple(outs)

    def forward(self, x, x1, scene_id=None, ann_id=None):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            x = torch.cat((x, x1[i]), dim=1)
            x = self.fusion_conv[i](x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class ResNetFusionGate(ResNet):

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNetFusionGate, self).__init__(depth=depth,
                                           in_channels=in_channels,
                                           stem_channels=stem_channels,
                                           base_channels=base_channels,
                                           num_stages=num_stages,
                                           strides=strides,
                                           dilations=dilations,
                                           out_indices=out_indices,
                                           style=style,
                                           deep_stem=deep_stem,
                                           avg_down=avg_down,
                                           frozen_stages=frozen_stages,
                                           conv_cfg=conv_cfg,
                                           norm_cfg=norm_cfg,
                                           norm_eval=norm_eval,
                                           dcn=dcn,
                                           stage_with_dcn=stage_with_dcn,
                                           plugins=plugins,
                                           with_cp=with_cp,
                                           zero_init_residual=zero_init_residual)
        self.fusion_conv = nn.ModuleList()
        for i in range(num_stages):
            self.fusion_conv.append(nn.Conv2d(
                base_channels * 2 ** (i + 2) * 2,
                base_channels * 2 ** (i + 2),
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False))
        self.fusion_layer = nn.ModuleList()
        for i in range(num_stages):
            self.fusion_layer.append(GateAttentionLayer(base_channels * 2 ** (i + 2)))

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.fusion_conv:
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

        for m in self.fusion_layer:
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x, x1, img_metas=None):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            x1_out = self.fusion_layer[i](x, x1[i])
            x = torch.cat((x, x1_out), dim=1)
            x = self.fusion_conv[i](x)
            draw = False
            if i == 0 and draw:
                dir = os.path.join('/home/qinran_2020/mmdetection_grasp/eval/gate_attention/')
                scene_dir = os.path.join(dir, 'scene_%04d' % img_metas[0]['sceneId'],
                                         img_metas[0]['camera'])
                # if not os.path.exists(scene_dir):
                #     os.makedirs(scene_dir)
                before_dir = os.path.join(scene_dir, 'before')
                after_dir = os.path.join(scene_dir, 'after')
                if not os.path.exists(before_dir):
                    os.makedirs(before_dir)
                if not os.path.exists(after_dir):
                    os.makedirs(after_dir)
                img_before_name = os.path.join(before_dir, '%04d.png' % img_metas[0]['annId'])
                img_after_name = os.path.join(after_dir, '%04d.png' % img_metas[0]['annId'])

                img_before = x1[0][0].permute(1, 2, 0).detach().cpu().numpy()
                img_before = np.mean(img_before, axis=2)
                heatmap_before = None
                heatmap_before = cv2.normalize(np.float32(img_before), heatmap_before, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_8U)
                heatmap_before = cv2.applyColorMap(heatmap_before, cv2.COLORMAP_JET)
                cv2.imwrite(img_before_name, heatmap_before)

                img_after = x1_out[0].permute(1, 2, 0).detach().cpu().numpy()
                img_after = np.mean(img_after, axis=2)
                heatmap_after = None
                heatmap_after = cv2.normalize(np.float32(img_after), heatmap_after, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                              dtype=cv2.CV_8U)
                heatmap_after = cv2.applyColorMap(heatmap_after, cv2.COLORMAP_JET)
                cv2.imwrite(img_after_name, heatmap_after)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)