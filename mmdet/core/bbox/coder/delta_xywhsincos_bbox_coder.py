import mmcv
import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder
from ..grasp_transforms import grasp_encode, grasp_decode

@BBOX_CODERS.register_module()
class DeltaXYWHSinCosBBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self,
                 target_means_hbb=(.0, .0, .0, .0),
                 target_stds_hbb=(1.0, 1.0, 1.0, 1.0),
                 target_means_obb=(.0, .0, .0, .0),
                 target_stds_obb=(1.0, 1.0, 1.0, 1.0)):
        super(BaseBBoxCoder, self).__init__()
        self.means_hbb = target_means_hbb
        self.stds_hbb = target_stds_hbb
        self.means_obb = target_means_obb
        self.stds_obb = target_stds_obb

    def encode(self, hbbox_pred, hbboxes, gt_hbboxes, gt_obboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        encode_hbboxes = bbox2delta(hbboxes, gt_hbboxes, self.means_hbb, self.stds_hbb)
        decode_hbboxes = delta2hbboxrec5(hbboxes, hbbox_pred, self.means_hbb, self.stds_hbb)
        encode_obboxes = rec2target(decode_hbboxes, gt_obboxes, self.means_obb, self.stds_obb)
        return encode_hbboxes, encode_obboxes

    def decode(self, anchors, bbox_pred, obb_pred):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        decode_hbboxes = delta2bbox(anchors, bbox_pred, self.means_hbb, self.stds_hbb)
        decode_obboxes = target2poly(decode_hbboxes, obb_pred, self.means_obb, self.stds_obb)
        return decode_obboxes


def bbox2delta(proposals, gt, means=(0, 0, 0, 0), stds=(1, 1, 1, 1)):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta2hbboxrec5(rois,
                   deltas,
                   means=(0, 0, 0, 0),
                   stds=(1, 1, 1, 1),
                   wh_ratio_clip=16/1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means  # 在bbox2delta中进行了标准化，这里要做逆变换
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    gtheta = gw.new_zeros((gw.size(0), gw.size(1)))
    rec = torch.stack([gx, gy, gw, gh, gtheta], dim=-1).view(deltas.size(0), -1)
    return rec

def rec2target(hbbox_rec, gt_rbbox_rec, means=(0, 0, 0, 0), stds=(1, 1, 1, 1)):
    hbbox_w = hbbox_rec[:, 2]
    hbbox_h = hbbox_rec[:, 3]
    hbbox_theta = hbbox_rec[:, 4]

    gt_rbbox_w = gt_rbbox_rec[:, 2]
    gt_rbbox_h = gt_rbbox_rec[:, 3]
    gt_rbbox_theta = gt_rbbox_rec[:, 4]

    delta_theta = gt_rbbox_theta - hbbox_theta

    delta_w = gt_rbbox_w / hbbox_w
    delta_h = gt_rbbox_h / hbbox_h

    # t11 = torch.cos(delta_theta) * delta_w
    # t12 = -torch.sin(delta_theta) * delta_h
    # t21 = torch.sin(delta_theta) * delta_w
    # t22 = torch.cos(delta_theta) * delta_h

    t11 = delta_w
    t12 = delta_h
    t21 = torch.sin(delta_theta)
    t22 = torch.cos(delta_theta)

    t = torch.stack([t11, t12, t21, t22], dim=-1)

    means = t.new_tensor(means).unsqueeze(0)
    stds = t.new_tensor(stds).unsqueeze(0)
    t = t.sub_(means).div_(stds)

    return t

def delta2bbox(rois,
               deltas,
               means=(0, 0, 0, 0),
               stds=(1, 1, 1, 1),
               max_shape=None):
    """
    Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.

    Args:
        rois (Tensor): boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): encoded offsets with respect to each roi.
            Has shape (N, 4). Note N = num_anchors * W * H when rois is a grid
            of anchors. Offset encoding follows [1]_.
        means (list): denormalizing means for delta coordinates
        stds (list): denormalizing standard deviation for delta coordinates
        max_shape (tuple[int, int]): maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): maximum aspect ratio for boxes.

    Returns:
        Tensor: boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.2817, 0.2817, 4.7183, 4.7183],
                [0.0000, 0.6321, 7.3891, 0.3679],
                [5.8967, 2.9251, 5.5033, 3.2749]])
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means # 在bbox2delta中进行了标准化，这里要做逆变换
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    bboxes = torch.stack([gx, gy, gw, gh], dim=-1).view_as(deltas)
    return bboxes



def target2poly(hbboxes,
                 obb_pred,
                 means=(0, 0, 0, 0),
                 stds=(1, 1, 1, 1)):
    means = obb_pred.new_tensor(means).repeat(1, obb_pred.size(1) // 4)
    stds = obb_pred.new_tensor(stds).repeat(1, obb_pred.size(1) // 4)
    deform_obb_pred = obb_pred * stds + means

    t11 = deform_obb_pred[:, 0::4]
    t12 = deform_obb_pred[:, 1::4]
    t21 = deform_obb_pred[:, 2::4]
    t22 = deform_obb_pred[:, 3::4]

    d11 = t11 * t22
    d12 = - t12 * t21
    d21 = t11 * t21
    d22 = t12 * t22

    # t11 = torch.cos(delta_theta) * delta_w
    # t12 = -torch.sin(delta_theta) * delta_h
    # t21 = torch.sin(delta_theta) * delta_w
    # t22 = torch.cos(delta_theta) * delta_h

    # t11 = delta_w
    # t12 = delta_h
    # t21 = torch.sin(delta_theta)
    # t22 = torch.cos(delta_theta)

    x_center = hbboxes[:, 0::4]
    y_center = hbboxes[:, 1::4]
    w = hbboxes[:, 2::4] - 1
    h = hbboxes[:, 3::4] - 1

    x1 = (-w / 2.0) * d11 + (-h / 2.0) * d12 + x_center
    y1 = (-w / 2.0) * d21 + (-h / 2.0) * d22 + y_center
    x2 = (w / 2.0) * d11 + (-h / 2.0) * d12 + x_center
    y2 = (w / 2.0) * d21 + (-h / 2.0) * d22 + y_center
    x3 = (w / 2.0) * d11 + (h / 2.0) * d12 + x_center
    y3 = (w / 2.0) * d21 + (h / 2.0) * d22 + y_center
    x4 = (-w / 2.0) * d11 + (h / 2.0) * d12 + x_center
    y4 = (-w / 2.0) * d21 + (h / 2.0) * d22 + y_center

    poly = torch.cat([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)
    if obb_pred.size(1) != 4:
        poly = poly.view(poly.size(0), 8, -1)
        poly = poly.permute(0, 2, 1)
        poly = poly.contiguous().view(poly.size(0), -1)
    return poly


