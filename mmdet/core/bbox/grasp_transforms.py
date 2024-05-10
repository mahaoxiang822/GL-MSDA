import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import torch.nn.init as init
import numpy as np
import cv2
import math
from graspnetAPI import RectGraspGroup
from graspnet.my_grasp import MyRectGraspGroup
import open3d as o3d
from math import *

def points_to_xywhtheta(points):
    """
    :param points: bs x n x 8 point array. Each line represents a grasp
    :return: label: bs x n x 5 label array: xc, yc, w, h, Theta
    """
    batch_size = points.size(0)
    label = torch.Tensor(batch_size, points.size(1), 5).type_as(points)
    label[:, :, 0] = (points[:, :, 0] + points[:, :, 4]) / 2
    label[:, :, 1] = (points[:, :, 1] + points[:, :, 5]) / 2
    label[:, :, 2] = torch.sqrt(torch.pow((points[:, :, 2] - points[:, :, 0]), 2)
                                + torch.pow((points[:, :, 3] - points[:, :, 1]), 2))
    label[:, :, 3] = torch.sqrt(torch.pow((points[:, :, 2] - points[:, :, 4]), 2)
                                + torch.pow((points[:, :, 3] - points[:, :, 5]), 2))
    label[:, :, 4] = - torch.atan((points[:, :, 3] - points[:, :, 1]) / (points[:, :, 2] - points[:, :, 0]))
    # label[:, :, 4] = label[:, :, 4] / np.pi * 180
    label[:, :, 4][label[:, :, 4] != label[:, :, 4]] = 0
    return label

def xywhtheta_to_points(label, max_shape=None):
    x = label[:,:,0:1]
    y = label[:,:,1:2]
    w = label[:,:,2:3]
    h = label[:,:,3:4]
    a = label[:,:,4:5]
    # a = a / 180 * np.pi
    vec1x = w/2*torch.cos(a) + h/2*torch.sin(a)
    vec1y = -w/2*torch.sin(a) + h/2*torch.cos(a)
    vec2x = w/2*torch.cos(a) - h/2*torch.sin(a)
    vec2y = -w/2*torch.sin(a) - h/2*torch.cos(a)
    x1 = x + vec1x
    y1 = y + vec1y
    x2 = x - vec2x
    y2 = y - vec2y
    x3 = x - vec1x
    y3 = y - vec1y
    x4 = x + vec2x
    y4 = y + vec2y
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
        x3 = x3.clamp(min=0, max=max_shape[1])
        y3 = y3.clamp(min=0, max=max_shape[0])
        x4 = x4.clamp(min=0, max=max_shape[1])
        y4 = y4.clamp(min=0, max=max_shape[0])
    return torch.cat([x1, y1, x2, y2, x3, y3, x4, y4], 2)

def grasp_encode(gt_grasps, anchors, angle_thresh, means, stds):
    if anchors.dim() == 2:
        anchors_widths = anchors[:, 2]
        anchors_heights = anchors[:, 3]
        anchors_ctr_x = anchors[:, 0]
        anchors_ctr_y = anchors[:, 1]
        anchors_angle = anchors[:, 4]

        gt_widths = gt_grasps[:, :, 2]
        gt_heights = gt_grasps[:, :, 3]
        gt_ctr_x = gt_grasps[:, :, 0]
        gt_ctr_y = gt_grasps[:, :, 1]
        gt_angle = gt_grasps[:, :, 4]

        targets_dx = (gt_ctr_x - anchors_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / anchors_widths.view(1,-1).expand_as(gt_ctr_x)
        targets_dy = (gt_ctr_y - anchors_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / anchors_heights.view(1,-1).expand_as(gt_ctr_x)
        targets_dw = torch.log(gt_widths / anchors_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / anchors_heights.view(1,-1).expand_as(gt_heights))
        # targets_da = torch.div(gt_angle - anchors_angle.view(1,-1).expand_as(gt_angle), angle_thresh)
        targets_da = gt_angle - anchors_angle.view(1, -1).expand_as(gt_angle)


    elif anchors.dim() == 3:
        anchors_widths = anchors[:, :, 2]
        anchors_heights = anchors[:,:, 3]
        anchors_ctr_x = anchors[:, :, 0]
        anchors_ctr_y = anchors[:, :, 1]
        anchors_angle = anchors[:, :, 4]

        gt_widths = gt_grasps[:, :, 2]
        gt_heights = gt_grasps[:, :, 3]
        gt_ctr_x = gt_grasps[:, :, 0]
        gt_ctr_y = gt_grasps[:, :, 1]
        gt_angle = gt_grasps[:, :, 4]

        targets_dx = (gt_ctr_x - anchors_ctr_x) / anchors_widths
        targets_dy = (gt_ctr_y - anchors_ctr_y) / anchors_heights
        targets_dw = torch.log(gt_widths / anchors_widths)
        targets_dh = torch.log(gt_heights / anchors_heights)
        # targets_da = torch.div(gt_angle - anchors_angle, angle_thresh)
        targets_da = gt_angle - anchors_angle
    else:
        raise ValueError('grasp_anchor input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_da), 2)

    targets = (targets - means.expand_as(targets).type_as(targets)) / stds.expand_as(targets).type_as(targets)

    return targets

def grasp_decode(deltas, anchors, angle_thresh, means, stds):
    means = means.expand_as(deltas).type_as(deltas)
    stds = stds.expand_as(deltas).type_as(deltas)
    deltas = deltas * stds + means

    if anchors.dim() == 2:
        anchors_widths = anchors[:, 2]
        anchors_heights = anchors[:, 3]
        anchors_ctr_x = anchors[:, 0]
        anchors_ctr_y = anchors[:, 1]
        anchors_angle = anchors[:, 4]

        deltas_widths = deltas[:, :, 2]
        deltas_heights = deltas[:, :, 3]
        deltas_ctr_x = deltas[:, :, 0]
        deltas_ctr_y = deltas[:, :, 1]
        deltas_angle = deltas[:, :, 4]

        targets_dx = deltas_ctr_x * anchors_widths.view(1,-1).expand_as(deltas_ctr_x) + anchors_ctr_x.view(1,-1).expand_as(deltas_ctr_x)
        targets_dy = deltas_ctr_y * anchors_heights.view(1,-1).expand_as(deltas_ctr_x)+ anchors_ctr_y.view(1,-1).expand_as(deltas_ctr_y)
        targets_dw = torch.exp(deltas_widths) * anchors_widths.view(1,-1).expand_as(deltas_widths)
        targets_dh = torch.exp(deltas_heights) * anchors_heights.view(1,-1).expand_as(deltas_heights)
        # targets_da = deltas_angle * angle_thresh + anchors_angle.view(1,-1).expand_as(deltas_angle)
        targets_da = deltas_angle + anchors_angle.view(1, -1).expand_as(deltas_angle)

    elif anchors.dim() == 3:
        anchors_widths = anchors[:, :, 2]
        anchors_heights = anchors[:,:, 3]
        anchors_ctr_x = anchors[:, :, 0]
        anchors_ctr_y = anchors[:, :, 1]
        anchors_angle = anchors[:, :, 4]

        deltas_widths = deltas[:, :, 2]
        deltas_heights = deltas[:, :, 3]
        deltas_ctr_x = deltas[:, :, 0]
        deltas_ctr_y = deltas[:, :, 1]
        deltas_angle = deltas[:, :, 4]

        targets_dx = deltas_ctr_x * anchors_widths + anchors_ctr_x
        targets_dy = deltas_ctr_y * anchors_heights + anchors_ctr_y
        targets_dw = torch.exp(deltas_widths) * anchors_widths
        targets_dh = torch.exp(deltas_heights) * anchors_heights
        # targets_da = deltas_angle * angle_thresh + anchors_angle
        targets_da = deltas_angle + anchors_angle

    else:
        raise ValueError('ref_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_da), 2)

    return targets

def grasp2result(labels, grasps, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if grasps.shape[0] == 0:
        return [np.zeros((0, 8), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(grasps, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            grasps = grasps.detach().cpu().numpy()
        return [grasps[labels == i, :] for i in range(num_classes)]



def points_to_xywhtheta_numpy(points):
    """
    :param points: bs x n x 8 point array. Each line represents a grasp
    :return: label: bs x n x 5 label array: xc, yc, w, h, Theta
    """
    label = np.zeros((points.shape[0], 5))
    label[:, 0] = (points[:, 0] + points[:, 4]) / 2
    label[:, 1] = (points[:, 1] + points[:, 5]) / 2
    label[:, 2] = np.sqrt(np.power((points[:, 2] - points[:, 0]), 2)
                          + np.power((points[:, 3] - points[:, 1]), 2))
    label[:, 3] = np.sqrt(np.power((points[:, 2] - points[:, 4]), 2)
                          + np.power((points[:, 3] - points[:, 5]), 2))
    label[:, 4] = np.arctan((points[:, 3] - points[:, 1]) / (points[:, 2] - points[:, 0]))
    # if (points[:, 2] - points[:, 0] == 0).any():
    #     import pdb
    #     pdb.set_trace()
    label[:, 4] = label[:, 4] / np.pi * 180
    return label

def jaccard_overlap_numpy(pred, gt):
    r1 = ((pred[0], pred[1]), (pred[2], pred[3]), pred[4])
    area_r1 = pred[2] * pred[3]
    r2 = ((gt[0], gt[1]), (gt[2], gt[3]), gt[4])
    area_r2 = gt[2] * gt[3]
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        ovr = int_area * 1.0 / (area_r1 + area_r2 - int_area)
        return ovr
    else:
        return 0

# 以下都是针对graspnet的
def rbbox_to_hbbox_list(rbboxes):
    if rbboxes is None:
        return None
    hbboxes = []
    for i in range(len(rbboxes)):
        rbbox = rbboxes[i]
        xs = rbbox.view(-1, 4, 2)[:, :, 0]
        ys = rbbox.view(-1, 4, 2)[:, :, 1]

        xmin = torch.min(xs, dim=1)[0]
        ymin = torch.min(ys, dim=1)[0]
        xmax = torch.max(xs, dim=1)[0]
        ymax = torch.max(ys, dim=1)[0]

        hbboxes.append(torch.stack([xmin, ymin, xmax, ymax], dim=1))

    return hbboxes

def points_to_xywhtheta_list(points):
    if points is None:
        return None
    xywhthetas= []
    for i in range(len(points)):
        x_center = (points[i][:, 0] + points[i][:, 4]) / 2
        y_center = (points[i][:, 1] + points[i][:, 5]) / 2
        w = torch.sqrt(torch.pow((points[i][:, 2] - points[i][:, 0]), 2)
                       + torch.pow((points[i][:, 3] - points[i][:, 1]), 2))
        h = torch.sqrt(torch.pow((points[i][:, 2] - points[i][:, 4]), 2)
                       + torch.pow((points[i][:, 3] - points[i][:, 5]), 2))
        # -pi/2, pi/2
        theta = torch.atan((points[i][:, 3] - points[i][:, 1]) / (points[i][:, 2] - points[i][:, 0]))
        xywhtheta = torch.stack([x_center, y_center, w, h, theta], dim=-1)
        xywhthetas.append(xywhtheta)
    return xywhthetas

def points_to_xywhtheta_graspnet(points):
    if points is None:
        return None
    x_center = (points[:, 0] + points[:, 4]) / 2
    y_center = (points[:, 1] + points[:, 5]) / 2
    w = torch.sqrt(torch.pow((points[:, 2] - points[:, 0]), 2)
                   + torch.pow((points[:, 3] - points[:, 1]), 2))
    h = torch.sqrt(torch.pow((points[:, 2] - points[:, 4]), 2)
                   + torch.pow((points[:, 3] - points[:, 5]), 2))
    theta = torch.atan((points[:, 3] - points[:, 1]) / (points[:, 2] - points[:, 0]))
    xywhtheta = torch.stack([x_center, y_center, w, h, theta], dim=-1)
    return xywhtheta

def points_to_xywhtheta_graspnet_np(points):
    if points is None:
        return None
    x_center = (points[:, 0] + points[:, 4]) / 2
    y_center = (points[:, 1] + points[:, 5]) / 2
    w = np.sqrt(np.power((points[:, 2] - points[:, 0]), 2)
                   + np.power((points[:, 3] - points[:, 1]), 2))
    h = np.sqrt(np.power((points[:, 2] - points[:, 4]), 2)
                   + np.power((points[:, 3] - points[:, 5]), 2))
    theta = np.arctan((points[:, 3] - points[:, 1]) / (points[:, 2] - points[:, 0]))
    xywhtheta = np.stack([x_center, y_center, w, h, theta], axis=1)
    return xywhtheta

def hbbox_to_xywhtheta(bboxes):
    if bboxes is None:
        return None
    num_bboxes = bboxes.size(0)
    w = (bboxes[..., 2] - bboxes[..., 0] + 1.0)
    h = (bboxes[..., 3] - bboxes[..., 1] + 1.0)
    x_center = (bboxes[..., 0] + bboxes[..., 2]) / 2.
    y_center = (bboxes[..., 1] + bboxes[..., 3]) / 2.
    theta = x_center.new_zeros(num_bboxes)
    rbboxes = torch.stack([x_center, y_center, w, h, theta], dim=1)
    return rbboxes


def xyxydepth_to_xywhthetadepth(bboxes):
    if bboxes is None:
        return None
    num_bboxes = bboxes.size(0)
    w = (bboxes[..., 2] - bboxes[..., 0] + 1.0)
    h = (bboxes[..., 3] - bboxes[..., 1] + 1.0)
    x_center = (bboxes[..., 0] + bboxes[..., 2]) / 2
    y_center = (bboxes[..., 1] + bboxes[..., 3]) / 2
    theta = x_center.new_zeros(num_bboxes)
    depth = bboxes[..., 4]
    rbboxes = torch.stack([x_center, y_center, w, h, theta, depth], dim=1)
    return rbboxes

def hbbox_to_xywhthetaz(bboxes, depth_map, img_shape):
    if bboxes is None:
        return None
    num_bboxes = bboxes.size(0)

    w = (bboxes[..., 2] - bboxes[..., 0] + 1.0)
    h = (bboxes[..., 3] - bboxes[..., 1] + 1.0)
    x_center = (bboxes[..., 0] + bboxes[..., 2]) / 2
    y_center = (bboxes[..., 1] + bboxes[..., 3]) / 2
    theta = x_center.new_zeros(num_bboxes)
    # mask = (x_center >= 0) & (x_center < depth_map.size(1)) & (y_center >= 0) & (y_center < depth_map.size(0))
    # x_center = x_center[mask]
    # y_center = y_center[mask]
    # w = w[mask]
    # h = h[mask]
    # theta = theta[mask]
    # x_center_index = x_center.new_zeros((len(x_center),))
    # y_center_index = y_center.new_zeros((len(x_center),))
    x_center_index = x_center.clamp(min=0, max=img_shape[1] - 1).long()
    y_center_index = y_center.clamp(min=0, max=img_shape[0] - 1).long()
    z = depth_map[y_center_index, x_center_index]
    # depth = torch.Tensor([depth_map[y_center_index[i], x_center_index[i]] for i in range(len(x_center))]).type_as(x_center)
    rbboxes = torch.stack([x_center, y_center, w, h, theta, z],dim=1)
    return rbboxes


def xywhtheta_to_rect_grasp_group(rbbox, score, img_shape):
    x_center = rbbox[:, 0].reshape(-1, 1)
    y_center = rbbox[:, 1].reshape(-1, 1)
    w = rbbox[:, 2].reshape(-1, 1)
    h = rbbox[:, 3].reshape(-1, 1)
    theta = rbbox[:, 4].reshape(-1, 1)
    axis = np.hstack((np.cos(theta), np.sin(theta)))
    open_point = np.hstack((x_center, y_center)) - w / 2 * axis
    object_id = np.zeros((len(x_center), 1))
    rect_grasp_group_array = np.hstack((x_center, y_center,
                                        open_point, h, score, object_id))
    valid_mask1 = rect_grasp_group_array[:, 0] <= img_shape[1] - 1
    valid_mask2 = rect_grasp_group_array[:, 0] >= 0
    valid_mask3 = rect_grasp_group_array[:, 1] >= 0
    valid_mask4 = rect_grasp_group_array[:, 1] <= img_shape[0] - 1
    valid_mask = np.logical_and(np.logical_and(valid_mask1, valid_mask2),
                                np.logical_and(valid_mask3, valid_mask4))
    rect_grasp_group_array = rect_grasp_group_array[valid_mask]
    # rect_grasp_group_array[:, 0] = np.clip(rect_grasp_group_array[:, 0], 0, img_shape[1] - 1)
    # rect_grasp_group_array[:, 1] = np.clip(rect_grasp_group_array[:, 1], 0, img_shape[0] - 1)
    rect_grasp_group = MyRectGraspGroup()
    rect_grasp_group.rect_grasp_group_array = rect_grasp_group_array
    return rect_grasp_group

def xywhthetadepth_to_rect_grasp_group(rbbox, score, img_shape, depth_map=None):
    x_center = rbbox[:, 0].reshape(-1, 1)
    y_center = rbbox[:, 1].reshape(-1, 1)
    w = rbbox[:, 2].reshape(-1, 1)
    h = rbbox[:, 3].reshape(-1, 1)
    theta = rbbox[:, 4].reshape(-1, 1)
    depth = rbbox[:, 5].reshape(-1, 1)
    axis = np.hstack((np.cos(theta), np.sin(theta)))
    open_point = np.hstack((x_center, y_center)) - w / 2 * axis
    object_id = np.zeros((len(x_center), 1))
    rect_grasp_group_array = np.hstack((x_center, y_center,
                                        open_point, h, score, object_id, depth))

    if depth_map is not None:
        valid_mask1 = rect_grasp_group_array[:, 0] <= img_shape[1] - 1
        valid_mask2 = rect_grasp_group_array[:, 0] >= 0
        valid_mask3 = rect_grasp_group_array[:, 1] >= 0
        valid_mask4 = rect_grasp_group_array[:, 1] <= img_shape[0] - 1
        valid_mask = np.logical_and(np.logical_and(valid_mask1, valid_mask2),
                                    np.logical_and(valid_mask3, valid_mask4))
        rect_grasp_group_array = rect_grasp_group_array[valid_mask]
        y = rect_grasp_group_array[:, 1].astype(np.int32)
        x = rect_grasp_group_array[:, 0].astype(np.int32)
        depth_mask = np.array([depth_map[y[i]][x[i]] > 0 for i in range(len(y))])
        rect_grasp_group_array = rect_grasp_group_array[depth_mask]

    # valid_mask1 = rect_grasp_group_array[:, 0] <= img_shape[1] - 1
    # valid_mask2 = rect_grasp_group_array[:, 0] >= 0
    # valid_mask3 = rect_grasp_group_array[:, 1] >= 0
    # valid_mask4 = rect_grasp_group_array[:, 1] <= img_shape[0] - 1
    # valid_mask = np.logical_and(np.logical_and(valid_mask1, valid_mask2),
    #                             np.logical_and(valid_mask3, valid_mask4))
    # rect_grasp_group_array = rect_grasp_group_array[valid_mask]
    # rect_grasp_group_array[:, 0] = np.clip(rect_grasp_group_array[:, 0], 0, img_shape[1] - 1)
    # rect_grasp_group_array[:, 1] = np.clip(rect_grasp_group_array[:, 1], 0, img_shape[0] - 1)
    rect_grasp_group = MyRectGraspGroup()
    rect_grasp_group.rect_grasp_group_array = rect_grasp_group_array
    return rect_grasp_group, depth_mask


def xywhthetadepthcls_to_rect_grasp_group(rbbox, score, depth):
    x_center = rbbox[:, 0].reshape(-1, 1)
    y_center = rbbox[:, 1].reshape(-1, 1)
    w = rbbox[:, 2].reshape(-1, 1)
    h = rbbox[:, 3].reshape(-1, 1)
    theta = rbbox[:, 4].reshape(-1, 1)
    axis = np.hstack((np.cos(theta), np.sin(theta)))
    open_point = np.hstack((x_center, y_center)) - w / 2 * axis
    object_id = np.zeros((len(x_center), 1))
    depth = depth.reshape(-1, 1)
    rect_grasp_group_array = np.hstack((x_center, y_center,
                                        open_point, h, score, object_id, depth))
    rect_grasp_group = MyRectGraspGroup()
    rect_grasp_group.rect_grasp_group_array = rect_grasp_group_array
    return rect_grasp_group


def xywhtheta_to_points_graspnet(rbbox, max_shape=None):
    x_center = rbbox[:, 0::5]
    y_center = rbbox[:, 1::5]
    w = rbbox[:, 2::5]
    h = rbbox[:, 3::5]
    theta = rbbox[:, 4::5]
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    x1 = costheta * (- w / 2.0) - sintheta * (- h / 2.0) + x_center
    y1 = sintheta * (-w / 2.0) + costheta * (- h / 2.0) + y_center
    x2 = costheta * (w / 2.0) - sintheta * (-h / 2.0) + x_center
    y2 = sintheta * (w / 2.0) + costheta * (-h / 2.0) + y_center
    x3 = costheta * (w / 2.0) - sintheta * (h / 2.0) + x_center
    y3 = sintheta * (w / 2.0) + costheta * (h / 2.0) + y_center
    x4 = costheta * (- w / 2.0) - sintheta * (h / 2.0) + x_center
    y4 = sintheta * (- w / 2.0) + costheta * (h / 2.0) + y_center

    if max_shape != None:
        poly = torch.cat([x1.clamp(min=0, max=max_shape[1] - 1),
                          y1.clamp(min=0, max=max_shape[0] - 1),
                          x2.clamp(min=0, max=max_shape[1] - 1),
                          y2.clamp(min=0, max=max_shape[0] - 1),
                          x3.clamp(min=0, max=max_shape[1] - 1),
                          y3.clamp(min=0, max=max_shape[0] - 1),
                          x4.clamp(min=0, max=max_shape[1] - 1),
                          y4.clamp(min=0, max=max_shape[0] - 1)], dim=-1)
    else:
        poly = torch.cat([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)
    if rbbox.size(1) != 5:
        poly = poly.view(poly.size(0), 8, -1)
        poly = poly.permute(0, 2, 1)
        poly = poly.contiguous().view(poly.size(0), -1)

    return poly

def xywhtheta_to_points_graspnet_np(rbbox):
    x_center = rbbox[:, 0::5]
    y_center = rbbox[:, 1::5]
    w = rbbox[:, 2::5]
    h = rbbox[:, 3::5]
    theta = rbbox[:, 4::5]
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    x1 = costheta * (- w / 2.0) - sintheta * (- h / 2.0) + x_center
    y1 = sintheta * (-w / 2.0) + costheta * (- h / 2.0) + y_center
    x2 = costheta * (w / 2.0) - sintheta * (-h / 2.0) + x_center
    y2 = sintheta * (w / 2.0) + costheta * (-h / 2.0) + y_center
    x3 = costheta * (w / 2.0) - sintheta * (h / 2.0) + x_center
    y3 = sintheta * (w / 2.0) + costheta * (h / 2.0) + y_center
    x4 = costheta * (- w / 2.0) - sintheta * (h / 2.0) + x_center
    y4 = sintheta * (- w / 2.0) + costheta * (h / 2.0) + y_center

    poly = np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)

    return poly

def batch_rect_average_depth(depths, centers, open_points, upper_points):
    results = []
    for i in range(len(centers)):
        axis1 = centers[i] - open_points[i]
        axis2 = upper_points[i] - centers[i]
        p1 = open_points[i] - axis2
        p2 = upper_points[i] - axis1
        p3 = upper_points[i] + axis1
        p4 = centers[i] + axis1 - axis2
        height = depths.shape[0]  # 原始图像高度
        width = depths.shape[1]  # 原始图像宽度
        widthRect = math.sqrt((p4[0] - p1[0]) ** 2 + (p4[1] - p1[1]) ** 2)  # 矩形框的宽度
        heightRect = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        angle = acos((p4[0] - p1[0]) / widthRect) * (180 / math.pi)  # 矩形框旋转角度
        if p4[1] < p1[1]:
            angle = -angle

        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

        rotateMat[0, 2] += (widthNew - width) / 2
        rotateMat[1, 2] += (heightNew - height) / 2
        depth_Rotation = cv2.warpAffine(depths, rotateMat, (widthNew, heightNew), borderValue=0)

        [[p1[0]], [p1[1]]] = np.dot(rotateMat, np.array([[p1[0]], [p1[1]], [1]]))
        [[p3[0]], [p3[1]]] = np.dot(rotateMat, np.array([[p3[0]], [p3[1]], [1]]))
        [[p2[0]], [p2[1]]] = np.dot(rotateMat, np.array([[p2[0]], [p2[1]], [1]]))
        [[p4[0]], [p4[1]]] = np.dot(rotateMat, np.array([[p4[0]], [p4[1]], [1]]))

        if p2[1] > p4[1]:
            p2[1], p4[1] = p4[1], p2[1]
        if p1[0] > p3[0]:
            p1[0], p3[0] = p3[0], p1[0]

        top = int(p2[1]) + int((p4[1] - p2[1]) / 3)
        down = int(p4[1]) - int((p4[1] - p2[1]) / 3)
        left = int(p1[0]) + int((p3[0] - p1[0]) / 3)
        right = int(p3[0]) - int((p3[1] - p1[1]) / 3)
        # left = int(p1[0])
        # right = int(p3[0])


        # depth_crop = depth_Rotation[int(p2[1]):int(p4[1]), int(p1[0]):int(p3[0])]
        # depth_avg = np.avg(depth_crop)
        depth_crop = depth_Rotation[top:down, left:right]
        depth_avg = np.mean(depth_crop[depth_crop > 0]) - 20
        results.append(depth_avg)
    results = np.array(results)
    return results

def xyxydepth2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois

# rrpn
def points_to_xyxy(polys):
    """
    without label
    :param polys: (x1, y1, ..., x4, y4) (n, 8)
    :return: boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    n = polys.shape[0]
    xs = np.reshape(polys, (n, 4, 2))[:, :, 0]
    ys = np.reshape(polys, (n, 4, 2))[:, :, 1]

    xmin = np.min(xs, axis=1)
    ymin = np.min(ys, axis=1)
    xmax = np.max(xs, axis=1)
    ymax = np.max(ys, axis=1)

    xmin = xmin[:, np.newaxis]
    ymin = ymin[:, np.newaxis]
    xmax = xmax[:, np.newaxis]
    ymax = ymax[:, np.newaxis]

    return np.concatenate((xmin, ymin, xmax, ymax), 1)

def points_to_rroi(rbbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, x, y, w, h, theta]
    """
    rrois_list = []
    for img_id, bboxes in enumerate(rbbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            bboxes = points_to_xywhtheta_graspnet(bboxes[:, :8])
            rrois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rrois = bboxes.new_zeros((0, 6))
        rrois_list.append(rrois)
    rrois = torch.cat(rrois_list, 0)
    return rrois