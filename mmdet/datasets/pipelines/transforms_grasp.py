import numpy as np
from numpy import random
import cv2

from ..builder import PIPELINES
import warnings
import copy

@PIPELINES.register_module()
class RandomCropKeepBoxes(object):
    def __call__(self, results):
        img = results['img']
        bboxes = results['gt_bboxes']
        bboxes_ignore = results['gt_bboxes_ignore']
        grasps = results['gt_grasps']
        height, width, _ = img.shape
        # Get the minimum boundary of all bboxes in dim x and y
        xmin = np.min(bboxes[:, 0])
        ymin = np.min(bboxes[:, 1])
        # Get the maximum boundary of all bboxes in dim x and y
        xmax = np.max(bboxes[:, 2])
        ymax = np.max(bboxes[:, 3])

        # Get top left corner's coordinate of the crop box
        x_start = int(random.uniform(0, xmin))
        y_start = int(random.uniform(0, ymin))
        # Get lower right corner corner's coordinate of the crop box
        x_end = int(random.uniform(xmax, width))
        y_end = int(random.uniform(ymax, height))

        # Crop the image
        img = img[y_start:y_end, x_start:x_end]
        img_shape = img.shape
        results['img_shape'] = img_shape
        results['img'] = img

        # Adjust the bboxes to fit the cropped image
        bboxes[:, 0::2] -= x_start
        bboxes[:, 1::2] -= y_start
        results['gt_bboxes'] = bboxes
        bboxes_ignore[:, 0::2] -= x_start
        bboxes_ignore[:, 1::2] -= y_start
        results['gt_bboxes_ignore'] = bboxes_ignore

        # Adjust the grasp boxes to fit the cropped image
        grasps[:, 0::2] -= x_start
        grasps[:, 1::2] -= y_start
        results['gt_grasps'] = grasps


        return results


@PIPELINES.register_module()
class ResizeToOrigin(object):
    def __init__(self,
                 keep_ratio):
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        h, w, _ = results['ori_shape']
        img = results['img']

        scale = np.array([h, w], dtype=np.float32) / img.shape[:2]
        if self.keep_ratio:
            min_scale = min(scale)
            max_scale = max(scale)
            scale = np.array([min_scale, min_scale], dtype=np.float32)
            h = int(scale[0] * img.shape[0])
            w = int(scale[1] * img.shape[1])
        img = cv2.resize(img, (w, h))
        results['img'] = img
        results['img_shape'] = img.shape
        # [h,w] -> [w,h]
        scale = scale[::-1]


        bboxes = results['gt_bboxes']
        ori_shape = bboxes.shape
        bboxes = (bboxes.reshape(-1, 2) * scale).reshape(ori_shape)
        results['gt_bboxes'] = bboxes

        bboxes_ignore = results['gt_bboxes_ignore']
        ori_shape = bboxes_ignore.shape
        bboxes_ignore = (bboxes_ignore.reshape(-1, 2) * scale).reshape(ori_shape)
        results['gt_bboxes_ignore'] = bboxes_ignore


        grasps = results['gt_grasps']
        ori_shape = grasps.shape
        grasps = (grasps.reshape(-1, 2) * scale).reshape(ori_shape)
        results['gt_grasps'] = grasps

        if scale[0] != scale[1]:
            warnings.warn("Resize scalers on x-axis and y-axis are not equal. Grasping rects is unusable.")

        return results


def rotcoords(coords, rot, w, h, isbbox=False):
    new_coords = np.zeros(coords.shape, dtype=np.float32)
    # (y, w-x)
    if rot == 1:
        new_coords[:, 0::2] = coords[:, 1::2]
        new_coords[:, 1::2] = w - coords[:, 0::2] - 1
    # (w-x, h-y)
    elif rot == 2:
        new_coords[:, 0::2] = w - coords[:, 0::2] - 1
        new_coords[:, 1::2] = h - coords[:, 1::2] - 1
    # (h-y,x)
    elif rot == 3:
        new_coords[:, 0::2] = h - coords[:, 1::2] - 1
        new_coords[:, 1::2] = coords[:, 0::2]
    if isbbox:
        new_coords = np.concatenate(
            (np.minimum(new_coords[:, 0:1], new_coords[:, 2:3]),
             np.minimum(new_coords[:, 1:2], new_coords[:, 3:4]),
             np.maximum(new_coords[:, 0:1], new_coords[:, 2:3]),
             np.maximum(new_coords[:, 1:2], new_coords[:, 3:4]))
            , axis=1)
    return new_coords

@PIPELINES.register_module()
class RandomVerticalRotate(object):
    def __init__(self,
                 rotate_ratio=0.5):
        self.rotate_ratio = rotate_ratio

    def __call__(self, results):
        img = results['img']
        bboxes = results['gt_bboxes']
        bboxes_ignore = results['gt_bboxes_ignore']
        grasps = results['gt_grasps']
        h, w, _ = img.shape
        r = np.random.rand()
        if r < self.rotate_ratio:
            return results
        r = random.randint(4)
        # 0: no rotation
        # 1: ccw 90 degrees
        # 2: ccw 180 degrees
        # 3: ccw 270 degrees
        if not r:
            return results
        img = np.rot90(img, k=r)
        results['img'] = img
        results['rotated_shape'] = img.shape
        results['img_shape'] = img.shape
        if bboxes is not None:
            bboxes = rotcoords(bboxes, r, w, h, isbbox=True)
            results['gt_bboxes'] = bboxes
        if bboxes_ignore is not None:
            bboxes_ignore = rotcoords(bboxes_ignore, r, w, h, isbbox=True)
            results['gt_bboxes_ignore'] = bboxes_ignore
        if grasps is not None:
            grasps = rotcoords(grasps, r, w, h, isbbox=False)
            results['gt_grasps'] = grasps
        return results



