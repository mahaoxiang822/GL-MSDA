import mmcv
import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder

@BBOX_CODERS.register_module()
class DeltaXYWHThetaGraspCoder(BaseBBoxCoder):
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
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_factor=180):
        super(BaseBBoxCoder, self).__init__()
        self.means = torch.FloatTensor(target_means)
        self.stds = torch.FloatTensor(target_stds)
        self.angle_factor = angle_factor

    def encode(self, grasps, gt_grasps):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        encoded_grasps = xywhtheta_to_delta(grasps, gt_grasps,
                                      self.means, self.stds)
        return encoded_grasps

    def decode(self, grasp_anchors, grasp_preds):
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

        decoded_grasps = delta_to_xywhtheta(grasp_preds, grasp_anchors,
                                      self.means, self.stds)

        return decoded_grasps

def xywhtheta_to_delta(rois, gts, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    x = rois[:, 0]
    y = rois[:, 1]
    w = rois[:, 2]
    h = rois[:, 3]
    theta = rois[:, 4]

    x_gt = gts[:, 0]
    y_gt = gts[:, 1]
    w_gt = gts[:, 2]
    h_gt = gts[:, 3]
    theta_gt = gts[:, 4]

    delta_x = (x_gt - x) / w
    delta_y = (y_gt - y) / h
    delta_w = torch.log(w_gt / w)
    delta_h = torch.log(h_gt / h)
    delta_theta = (theta_gt - theta)

    deltas = torch.stack([delta_x, delta_y, delta_w, delta_h, delta_theta], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta_to_xywhtheta(deltas,
                       roi_recs,
                       means=[0, 0, 0, 0, 0],
                       stds=[1, 1, 1, 1, 1]):
    '''
            :param deltas: (tx,ty,tw,th,ttheta)
            :param roi_recs: (x,y,w,h,theta)
            :return:
                roi_preds:(x_pred,y_pred,w_pred,h_pred,theta_pred)
            '''
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    deform_deltas = deltas * stds + means

    delta_x = deform_deltas[:, 0::5]
    delta_y = deform_deltas[:, 1::5]
    delta_w = deform_deltas[:, 2::5]
    delta_h = deform_deltas[:, 3::5]
    delta_theta = deform_deltas[:, 4::5]

    x = (roi_recs[:, 0]).unsqueeze(1).expand_as(delta_x)
    y = (roi_recs[:, 1]).unsqueeze(1).expand_as(delta_y)
    w = (roi_recs[:, 2]).unsqueeze(1).expand_as(delta_w)
    h = (roi_recs[:, 3]).unsqueeze(1).expand_as(delta_h)
    theta = (roi_recs[:, 4]).unsqueeze(1).expand_as(delta_theta)

    x_pred = delta_x * w + x
    y_pred = delta_y * h + y
    w_pred = w * delta_w.exp()
    h_pred = h * delta_h.exp()
    theta_pred = theta + delta_theta
    return torch.stack([x_pred, y_pred, w_pred, h_pred, theta_pred], dim=-1).view_as(deltas)


