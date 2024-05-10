from mmdet.datasets.vmrd import VMRDDataset
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import shutil
from xml.etree.ElementTree import ElementTree, Element

ann_file = 'data/vmrd/ImageSets/Main/trainval.txt'
ann_write_file = 'data/vmrd/ImageSets/Main/aug_trainval.txt'
img_prefix = 'data/vmrd/'
img_dir = 'data/vmrd/JPEGImages/'
anno_dir = 'data/vmrd/Annotations/'
img_write_dir = 'data/vmrd/aug_trainval_JPEGImages/'
anno_write_dir = 'data/vmrd/aug_trainval_Annotations/'
grasp_dir = 'data/vmrd/Grasps/'
grasp_write_dir = 'data/vmrd/aug_trainval_Grasps/'


def rotcoords(coords, rot, w, h, isbbox=False):
    new_coords = np.zeros(coords.shape)
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
        new_coords = new_coords + 1
    return new_coords

def getRotatedImg(r, img_path, img_write_path):
    img = cv2.imread(img_path)
    if r == 0:
        cv2.imwrite(img_write_path, img)
    # rows, cols = img.shape[:2]
    # a, b = cols / 2, rows / 2
    # M = cv2.getRotationMatrix2D((a, b), angle, 1)
    # if angle == 180:
    #     new_cols = cols
    #     new_rows = rows
    # else:
    #     new_cols = rows
    #     new_rows = cols
    # rotated_img = cv2.warpAffine(img, M, (new_cols, new_rows))  # 旋转后的图像保持大小不变
    # cv2.imwrite(img_write_path, rotated_img)
    else:
        rotate_img = np.rot90(img, k=r)
        cv2.imwrite(img_write_path, rotate_img)



def getRotatedAnno(r, id_rotate, anno_path, anno_write_path, width, height):
    tree = ET.parse(anno_path)
    root = tree.getroot()
    root.find('filename').text = id_rotate + '.JPG'
    if r == 0:
        tree.write(anno_write_path)
        return
    objects = root.findall("object")
    if r == 1 or r == 3:
        width_rotate = height
        height_rotate = width
    else:
        width_rotate = width
        height_rotate = height
    root.find('size').find('width').text = str(int(width_rotate))
    root.find('size').find('height').text = str(int(height_rotate))
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        bbox_coords = np.array([x1, y1, x2, y2]).reshape(-1, 4)
        bbox_rotate = rotcoords(bbox_coords, r, width, height, isbbox=True).reshape(4)

        bbox.find('xmin').text = str(int(bbox_rotate[0]))
        bbox.find('ymin').text = str(int(bbox_rotate[1]))
        bbox.find('xmax').text = str(int(bbox_rotate[2]))
        bbox.find('ymax').text = str(int(bbox_rotate[3]))

    tree.write(anno_write_path)  # 保存修改后的XML文件

def getRotatedGraspAnno(r, grasp_path, grasp_write_path, width, height):
    with open(grasp_path) as f_in:
        grasps = [x.strip() for x in f_in.readlines()]
    with open(grasp_write_path, mode='w') as f_out:
        grasp_bbox = np.array([grasp.split(' ')[:8] for grasp in grasps], dtype=np.float32)
        grasp_ind = np.array([grasp.split(' ')[8] for grasp in grasps], dtype=np.float32)
        grasp_str = [grasp.split(' ')[9] for grasp in grasps]
        if r == 0:
            grasp_bbox_rotate = grasp_bbox
        else:
            grasp_bbox_rotate = rotcoords(grasp_bbox, r, width, height, isbbox=False)
        for i in range(len(grasp_bbox_rotate)):
            outline = ' '.join(map(str, grasp_bbox_rotate[i])) + ' ' \
                      + str(int(grasp_ind[i])) + ' '\
                      + grasp_str[i] + '\n'
            f_out.write(outline)

def rotate_aug(data_infos):
    num_images = len(data_infos)
    print('before rotate: ' + str(int(num_images)))
    rotate_data_infos = []
    # totally 3 rotation angles
    for i in range(num_images):
        for r in [0, 1, 2, 3]:
            # 逆时针
            # print("num_images:", i)
            width = data_infos[i]['width']
            height = data_infos[i]['height']
            id = data_infos[i]['id']
            id_rotate = data_infos[i]['id'] + '_' + str(r)
            img_path = os.path.join(img_dir, id + '.jpg')
            img_write_path = os.path.join(img_write_dir, id_rotate + '.jpg')
            anno_path = os.path.join(anno_dir, id + '.xml')
            anno_write_path = os.path.join(anno_write_dir, id_rotate + '.xml')
            grasp_path = os.path.join(grasp_dir, id + '.txt')
            grasp_write_path = os.path.join(grasp_write_dir, id_rotate + '.txt')

            info = {}
            info['id'] = id_rotate
            info['filename'] = 'aug_trainval_JPEGImages/' + id_rotate + '.jpg'
            info['width'] = width if r == 2 or r == 0 else height
            info['height'] = height if r == 2 or r == 0 else width

            getRotatedImg(r, img_path, img_write_path)

            getRotatedAnno(r, id_rotate, anno_path, anno_write_path, width, height)

            getRotatedGraspAnno(r, grasp_path, grasp_write_path, width, height)

            rotate_data_infos.append(info)
    return rotate_data_infos

def HorizontalFlipImg(img_path, img_write_path):
    img = cv2.imread(img_path)
    mirror_img = cv2.flip(img, 1)
    cv2.imwrite(img_write_path, mirror_img)

def HorizontalFlipAnno(id_flip, width, anno_path, anno_write_path):
    tree = ET.parse(anno_path)
    root = tree.getroot()
    root.find('filename').text = id_flip + '.JPG'
    objects = root.findall("object")
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        x1 = width - x1
        x2 = width - x2

        assert x1>0
        assert x2>0

        bbox.find('xmin').text=str(int(x2))
        bbox.find('xmax').text=str(int(x1))

    tree.write(anno_write_path)  # 保存修改后的XML文件

def HorizontalFlipGraspAnno(grasp_path, grasp_write_path, width):
    with open(grasp_path) as f_in:
        grasps = [x.strip() for x in f_in.readlines()]
    with open(grasp_write_path, mode='w') as f_out:
        grasp_bbox = np.array([grasp.split(' ')[:8] for grasp in grasps], dtype=np.float32)
        grasp_ind = np.array([grasp.split(' ')[8] for grasp in grasps], dtype=np.float32)
        grasp_str = [grasp.split(' ')[9] for grasp in grasps]
        flip_grasp_bbox = np.zeros(grasp_bbox.shape)
        flip_grasp_bbox[:, 0] = width - grasp_bbox[:, 2] - 1
        flip_grasp_bbox[:, 1] = grasp_bbox[:, 3]
        flip_grasp_bbox[:, 2] = width - grasp_bbox[:, 0] - 1
        flip_grasp_bbox[:, 3] = grasp_bbox[:, 1]
        flip_grasp_bbox[:, 4] = width - grasp_bbox[:, 6] - 1
        flip_grasp_bbox[:, 5] = grasp_bbox[:, 7]
        flip_grasp_bbox[:, 6] = width - grasp_bbox[:, 4] - 1
        flip_grasp_bbox[:, 7] = grasp_bbox[:, 5]
        for i in range(len(flip_grasp_bbox)):
            outline = ' '.join(map(str, flip_grasp_bbox[i])) + ' ' \
                      + str(int(grasp_ind[i])) + ' ' \
                      + grasp_str[i] + '\n'
            f_out.write(outline)

def horizontal_filp_aug(data_infos):
    num_images = len(data_infos)
    print("before flip: " + str(int(num_images)))
    flip_data_infos = []
    for i in range(num_images):
        width = data_infos[i]['width']
        height = data_infos[i]['height']
        id = data_infos[i]['id']
        id_flip = data_infos[i]['id'] + '__' + str(1)
        img_path = os.path.join(img_write_dir, id + '.jpg')
        img_write_path = os.path.join(img_write_dir, id_flip + '.jpg')
        anno_path = os.path.join(anno_write_dir, id + '.xml')
        anno_write_path = os.path.join(anno_write_dir, id_flip + '.xml')
        grasp_path = os.path.join(grasp_write_dir, id + '.txt')
        grasp_write_path = os.path.join(grasp_write_dir, id_flip + '.txt')


        info = {}
        info['id'] = id_flip
        info['filename'] = 'aug_trainval_JPEGImages/' + id_flip + '.jpg'
        info['width'] = width
        info['height'] = height

        HorizontalFlipImg(img_path, img_write_path)
        HorizontalFlipAnno(id_flip, width, anno_path, anno_write_path)
        HorizontalFlipGraspAnno(grasp_path, grasp_write_path, width)
        flip_data_infos.append(info)
    data_infos.extend(flip_data_infos)
    return data_infos

def write_ann_file(ann_file, data_infos):
    with open(ann_file, mode='w') as f:
        for i in range(len(data_infos)):
            id = data_infos[i]['id']
            outline = id + '\n'
            f.write(outline)


if __name__ == '__main__':
    VMRD = VMRDDataset(ann_file=ann_file,
                   pipeline=None,
                   img_prefix=img_prefix)
    data_infos = VMRD.load_annotations(ann_file)

    if not os.path.exists(img_write_dir):
        os.makedirs(img_write_dir)
    if not os.path.exists(anno_write_dir):
        os.makedirs(anno_write_dir)
    if not os.path.exists(grasp_write_dir):
        os.makedirs(grasp_write_dir)


    rotate_data_infos = rotate_aug(data_infos)
    flip_data_infos = horizontal_filp_aug(rotate_data_infos)
    print("after aug: " + str(len(flip_data_infos)))
    write_ann_file(ann_write_file, flip_data_infos)

