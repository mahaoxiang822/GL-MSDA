import os
import cv2
import json
from tqdm import tqdm

classes = ('grasp', )

def load_grasps(filename):
    f = open(filename, 'r')
    grasps = []
    for line in f.readlines():
        temp = line.split(';')
        grasp = [float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4])]
        grasps.append(grasp)
    return grasps

def convert_to_coco_debug(src_path, dst_ann_train):
    image_dir = os.path.join(src_path, 'rgb')
    grasp_dir = os.path.join(src_path, 'grasp')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(classes):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    img_list = os.listdir(image_dir)

    with open(dst_ann_train, 'w') as f_out:
        inst_count = 1
        image_id = 1
        img_name = img_list[0]
        id = img_name.split('.')[0][:-4]
        img = cv2.imread(os.path.join(image_dir, img_name))
        height, width, c = img.shape

        single_image = {}
        single_image['file_name'] = img_name
        single_image['id'] = image_id
        single_image['width'] = width
        single_image['height'] = height
        data_dict['images'].append(single_image)

        # annotations
        grasps = load_grasps(os.path.join(grasp_dir, id + '_grasps.txt'))
        for grasp in grasps:
            single_obj = {}
            single_obj['category_id'] = 1
            single_obj['iscrowd'] = 0
            single_obj['bbox'] = grasp[0], grasp[1], grasp[2], grasp[3], grasp[4]
            single_obj['image_id'] = image_id
            single_obj['id'] = inst_count
            data_dict['annotations'].append(single_obj)
            inst_count = inst_count + 1
        image_id = image_id + 1
        json.dump(data_dict, f_out)


def convert_to_coco(src_path, dst_ann_train, dst_ann_test):
    image_dir = os.path.join(src_path, 'rgb')
    grasp_dir = os.path.join(src_path, 'grasp')


    data_dict_train = {}
    data_dict_train['images'] = []
    data_dict_train['categories'] = []
    data_dict_train['annotations'] = []
    for idex, name in enumerate(classes):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict_train['categories'].append(single_cat)

    img_list = os.listdir(image_dir)
    id_list = [img[2:-8] for img in img_list]
    sorted_id_list = sorted(enumerate(id_list), key=lambda x: x[1])
    sorted_index = [sorted_id[0] for sorted_id in sorted_id_list]
    sorted_img_list = [img_list[id] for id in sorted_index]
    train_num = int(4 * len(img_list) / 5)
    test_num = len(img_list) - train_num

    with open(dst_ann_train, 'w') as f_out:
        inst_count = 1
        image_id = 1
        for i in tqdm(range(train_num), 'img_name:' + sorted_img_list[image_id - 1]):
            # print(i)
            img_name = sorted_img_list[i]
            id = img_name.split('.')[0][:-4]
            img = cv2.imread(os.path.join(image_dir, img_name))
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = img_name
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict_train['images'].append(single_image)

            # annotations
            grasps = load_grasps(os.path.join(grasp_dir, id + '_grasps.txt'))
            for grasp in grasps:
                single_obj = {}
                single_obj['category_id'] = 1
                single_obj['iscrowd'] = 0
                single_obj['bbox'] = grasp[0], grasp[1], grasp[2], grasp[3], grasp[4]
                single_obj['image_id'] = image_id
                single_obj['id'] = inst_count
                data_dict_train['annotations'].append(single_obj)
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict_train, f_out)

    data_dict_test = {}
    data_dict_test['images'] = []
    data_dict_test['categories'] = []
    data_dict_test['annotations'] = []
    for idex, name in enumerate(classes):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict_test['categories'].append(single_cat)

    with open(dst_ann_test, 'w') as f_out:
        inst_count = 1
        image_id = 1
        for i in tqdm(range(test_num), 'img_name:' + sorted_img_list[image_id - 1 + train_num]):
            # print(i+train_num)
            img_name = sorted_img_list[i + train_num]
            id = img_name.split('.')[0][:-4]
            img = cv2.imread(os.path.join(image_dir, img_name))
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = img_name
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict_test['images'].append(single_image)

            # annotations
            grasps = load_grasps(os.path.join(grasp_dir, id + '_grasps.txt'))
            for grasp in grasps:
                single_obj = {}
                single_obj['category_id'] = 1
                single_obj['iscrowd'] = 0
                single_obj['bbox'] = grasp[0], grasp[1], grasp[2], grasp[3], grasp[4]
                single_obj['image_id'] = image_id
                single_obj['id'] = inst_count
                data_dict_test['annotations'].append(single_obj)
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict_test, f_out)


if __name__ == '__main__':
    from multiprocessing import Pool
    p = Pool(processes=12)
    p.apply_async(convert_to_coco, (r'/home/qinran_2020/data/jacquard',
                    r'/home/qinran_2020/data/jacquard/annotations/train.json',
                    r'/home/qinran_2020/data/jacquard/annotations/test.json'))
    p.close()
    p.join()
    # convert_to_coco_debug(r'/home/qinran_2020/data/jacquard',
    #                 r'/home/qinran_2020/data/jacquard/annotations/debug_train.json')