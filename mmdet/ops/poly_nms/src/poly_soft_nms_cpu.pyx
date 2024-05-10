# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# Modified by Kai Chen
# ----------------------------------------------------------

# cython: language_level=3, boundscheck=False

import numpy as np
cimport numpy as np
import cv2

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b



def poly_soft_nms_cpu(
    np.ndarray[float, ndim=2] boxes_in,
    float iou_thr,
    unsigned int method=1,
    float sigma=0.5,
    float min_score=0.001,
):
    boxes = boxes_in.copy()
    cdef int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1, x2, y1, y2, tx1, tx2, ty1, ty2, tx3, ty3, tx4, ty4, ts, area, weight, ov
    cdef np.ndarray[float, ndim=3] order_points
    cdef float inter_area, iou
    inds = np.arange(N)

    for i in range(N):
        maxscore = boxes[i, 8]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        tx3 = boxes[i, 4]
        ty3 = boxes[i, 5]
        tx4 = boxes[i, 6]
        ty4 = boxes[i, 7]
        ts = boxes[i, 8]
        ti = inds[i]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 8]:
                maxscore = boxes[pos, 8]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]
        boxes[i, 5] = boxes[maxpos, 5]
        boxes[i, 6] = boxes[maxpos, 6]
        boxes[i, 7] = boxes[maxpos, 7]
        boxes[i, 8] = boxes[maxpos, 8]
        inds[i] = inds[maxpos]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = tx3
        boxes[maxpos, 5] = ty3
        boxes[maxpos, 6] = tx4
        boxes[maxpos, 7] = ty4
        boxes[maxpos, 8] = ts

        inds[maxpos] = ti

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        tx3 = boxes[i, 4]
        ty3 = boxes[i, 5]
        tx4 = boxes[i, 6]
        ty4 = boxes[i, 7]
        ts = boxes[i, 8]

        tcnt = np.array([[tx1, ty1], [tx2, ty2], [tx3, ty3], [tx4, ty4]])
        trect = cv2.minAreaRect(tcnt)
        tarea = trect[1][0] * trect[1][1]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below
        # threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            x3 = boxes[pos, 4]
            y3 = boxes[pos, 5]
            x4 = boxes[pos, 6]
            y4 = boxes[pos, 7]
            s = boxes[pos, 8]

            cnt = np.array([[x1, y1], [x2, y2], [x3, y4], [x4, y4]])
            rect = cv2.minAreaRect(cnt)
            area = rect[1][0] * rect[1][1]
            inter_points = cv2.rotatedRectangleIntersection(trect, rect)[1]
            if inter_points is not None:
                order_points = cv2.convexHull(inter_points, returnPoints=True)
                inter_area = cv2.contourArea(order_points)
                iou = inter_area * 1.0 / (area + tarea - inter_area)
                if method == 1:  # linear
                    if iou > iou_thr:
                        weight = 1 - iou
                    else:
                        weight = 1
                elif method == 2:  # gaussian
                    weight = np.exp(-(iou * iou) / sigma)
                else:  # original NMS
                    if iou > iou_thr:
                        weight = 0
                    else:
                        weight = 1
                boxes[pos, 8] = weight * boxes[pos, 8]

                # if box score falls below threshold, discard the box by
                # swapping with last box update N
                if boxes[pos, 8] < min_score:
                    boxes[pos, 0] = boxes[N-1, 0]
                    boxes[pos, 1] = boxes[N-1, 1]
                    boxes[pos, 2] = boxes[N-1, 2]
                    boxes[pos, 3] = boxes[N-1, 3]
                    boxes[pos, 4] = boxes[N-1, 4]
                    boxes[pos, 5] = boxes[N-1, 5]
                    boxes[pos, 6] = boxes[N-1, 6]
                    boxes[pos, 7] = boxes[N-1, 7]
                    boxes[pos,  8] = boxes[N-1, 8]
                    inds[pos] = inds[N - 1]
                    N = N - 1
                    pos = pos - 1
            pos = pos + 1


    return boxes[:N], inds[:N]
