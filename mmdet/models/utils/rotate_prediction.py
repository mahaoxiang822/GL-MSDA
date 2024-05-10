from torch import nn
from ..builder import build_loss
import torch
import torch.nn.functional as F

class RotatePrediction(nn.Module):

    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_classes,
                 loss_rgb=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_depth=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.1)):

        super(RotatePrediction, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes

        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.feat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(inplace=True),
        )
        self.rgb_fc1 = nn.Sequential(
            nn.Linear(self.feat_channels * 7 * 7, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.rgb_fc2 = nn.Linear(self.feat_channels, self.num_classes)

        self.depth_conv = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.feat_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(inplace=True)
        )
        self.depth_fc1 = nn.Sequential(
            nn.Linear(self.feat_channels * 7 * 7, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.depth_fc2 = nn.Linear(self.feat_channels, self.num_classes)

        self.loss_rgb = build_loss(loss_rgb)
        self.loss_depth = build_loss(loss_depth)

    def forward_train(self, x1_rgb, x2_rgb, x1_depth, x2_depth, rgb_labels, depth_labels):
        x1_rgb = self.adaptive_pool(x1_rgb)
        x2_rgb = self.adaptive_pool(x2_rgb)
        x_rgb = torch.cat([x1_rgb, x2_rgb], dim=1)
        x_rgb = self.rgb_conv(x_rgb)
        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_rgb = self.rgb_fc1(x_rgb)
        x_rgb = self.rgb_fc2(x_rgb)
        x_rgb_weights = x_rgb.new_ones(x_rgb.shape[0], 1)
        rgb_labels = rgb_labels.long()
        loss_rgb = self.loss_rgb(x_rgb, rgb_labels, x_rgb_weights, avg_factor=x_rgb_weights.size(0))

        x1_depth = self.adaptive_pool(x1_depth)
        x2_depth = self.adaptive_pool(x2_depth)
        x_depth = torch.cat([x1_depth, x2_depth], dim=1)
        x_depth = self.depth_conv(x_depth)
        x_depth = x_depth.view(x_depth.size(0), -1)
        x_depth = self.depth_fc1(x_depth)
        x_depth = self.depth_fc2(x_depth)
        x_depth_weights = x_depth.new_ones(x_depth.shape[0], 1)
        depth_labels = depth_labels.long()
        loss_depth = self.loss_depth(x_depth, depth_labels, x_depth_weights, avg_factor=x_depth_weights.size(0))

        loss = dict(loss_rgb_rotation=loss_rgb,
                    loss_depth_rotation=loss_depth)
        return loss