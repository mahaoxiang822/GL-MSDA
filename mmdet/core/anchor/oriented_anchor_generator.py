import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from .builder import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register_module()
class OrientedAnchorGenerator(object):
    """Standard anchor generator for 2D anchor-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_anchors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_anchors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales,
                 angles):
        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = min(strides)

        self.scales = np.array(scales)
        self.ratios = np.array(ratios)
        self.angles = np.array(angles)

        self.base_anchors = torch.from_numpy(self.generate_base_anchors(self.base_sizes,
                                                               self.ratios,
                                                               self.scales,
                                                               self.angles))
        self.num_base_anchors = self.base_anchors.size(0)
        # [x1, y1, x2, y2] -> [xc, yc, w, h]
        self.base_anchors = torch.cat([
            0 * self.base_anchors[:, 0:1],
            0 * self.base_anchors[:, 1:2],
            self.base_anchors[:, 2:3] - self.base_anchors[:, 0:1] + 1,
            self.base_anchors[:, 3:4] - self.base_anchors[:, 1:2] + 1,
            self.base_anchors[:, 4:5]
        ], dim=1)
        pass

    def generate_anchors(self, feat_height, feat_width, rois):
        # roi局部anchor，是局部坐标（原点为roi左上角）
        # feat stride x, dim: bs x N x 1
        fsx = ((rois[:, :, 3:4] - rois[:, :, 1:2]) / feat_width).data.cpu().numpy()
        # feat stride y, dim: bs x N x 1
        fsy = ((rois[:, :, 4:5] - rois[:, :, 2:3]) / feat_height).data.cpu().numpy()

        # bs x N x W, center point of each cell
        shift_x = np.arange(0, feat_width) * fsx + fsx / 2
        # bs x N x H, center point of each cell
        shift_y = np.arange(0, feat_height) * fsy + fsy / 2

        # [bs x N x W x H (x coords), bs x N x W x H (y coords)]
        shift_x, shift_y = (
            np.repeat(np.expand_dims(shift_x, 2), shift_y.shape[2], axis=2),
            np.repeat(np.expand_dims(shift_y, 3), shift_x.shape[2], axis=3)
        )
        # bs x N x W*H x 2
        shifts = torch.cat([torch.from_numpy(shift_x).unsqueeze(4), torch.from_numpy(shift_y).unsqueeze(4)],4)
        shifts = shifts.contiguous().view(rois.size(0), rois.size(1), -1, 2)
        # bs x N x W*H x 5
        shifts = torch.cat([
            shifts,
            torch.zeros(shifts.size()[:-1] + (3,)).type_as(shifts)
        ], dim=-1)
        shifts = shifts.contiguous().float()

        A = self.num_base_anchors
        # K = W*H
        K = shifts.size(-2)

        # anchors = self._anchors.view(1, A, 5) + shifts.view(1, K, 5).permute(1, 0, 2).contiguous()

        anchors = self.base_anchors.view(1, A, 5) + shifts.unsqueeze(-2)
        # bs*N x K*A x 5
        anchors = anchors.view(rois.size(0) * rois.size(1) , K * A, 5)

        return anchors


    def generate_base_anchors(self,
                                  base_size=32,
                                  ratios=np.array([1]),
                                  scales=np.array([54. / 32.]),
                                  angles=30 * np.arange(6) - 75):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales wrt a reference (0, 0, 15, 15) window.
        """
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = self.ratio_enum(base_anchor, ratios)
        vertical_anchors = np.vstack([self.scale_enum(ratio_anchors[i, :], scales)
                                      for i in range(ratio_anchors.shape[0])])
        anchors = np.vstack([self.angle_enum(vertical_anchors[i, :], angles)
                             for i in range(vertical_anchors.shape[0])])
        return anchors

    def ratio_enum(self, anchor, ratios):
        """
        Enumerate a set of anchors for each aspect ratio wrt an anchor.
        """

        w, h, x_ctr, y_ctr = self.whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = self.mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def scale_enum(self, anchor, scales):
        """
        Enumerate a set of anchors for each scale wrt an anchor.
        """

        w, h, x_ctr, y_ctr = self.whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = self.mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def angle_enum(self, anchor, angles):

        anchors = np.hstack((anchor, np.array([0])))
        anchors = np.repeat(np.expand_dims(anchors, 0), len(angles), axis=0)
        anchors[:, -1] = angles / 180 * np.pi
        return anchors

    def whctrs(self, anchor):
        """
        Return width, height, x center, and y center for an anchor (window).
        """

        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def mkanchors(self, ws, hs, x_ctr, y_ctr):
        """
        Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """

        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
        return anchors

