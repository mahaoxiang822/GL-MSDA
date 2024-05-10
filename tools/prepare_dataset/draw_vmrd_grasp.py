from mmdet.datasets import VMRDDataset
import os
import cv2

ann_file = 'data/aug_vmrd/ImageSets/Main/aug_trainval_debug.txt'
img_prefix = 'data/aug_vmrd/'
output_dir = 'data/aug_vmrd/draw/'
VMRD = VMRDDataset(ann_file=ann_file,
                      pipeline=None,
                      img_prefix=img_prefix)
data_infos = VMRD.load_annotations(ann_file)
for i in range(len(data_infos)):
    id = data_infos[i]['id']
    img_path = os.path.join(img_prefix, data_infos[i]['filename'])
    img = cv2.imread(img_path)

    out_path = os.path.join(output_dir, data_infos[i]['id'] + '.jpg')

    grasps_anno = VMRD._load_grasp_annotation(id)
    gt_grasps = grasps_anno['grasps']
    gt_grasp_inds = grasps_anno['grasp_inds']
    bboxes_anno = VMRD._load_vmrd_annotation(id)
    labels = bboxes_anno['labels']
    for j in range(len(gt_grasps)):
        if gt_grasp_inds[j] < 1 or gt_grasp_inds[j] >= len(labels) + 1:
            continue
        cls = VMRD.CLASSES[labels[gt_grasp_inds[j] - 1]]
        bbox_color = VMRD.color_dict[cls]
        img = VMRD.draw_single_grasp(img, gt_grasps[j], cls, bbox_color)

    cv2.imwrite(out_path, img)






