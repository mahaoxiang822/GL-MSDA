from .builder import DATASETS
from graspnetAPI import GraspNet
from .custom import CustomDataset
import os
import numpy as np
from .pipelines import Compose
from PIL import Image
from graspnet.my_graspnet_eval import MyGraspNetEval
from collections import OrderedDict
from mmcv.utils import print_log

TOTAL_SCENE_NUM = 500
GRASP_HEIGHT = 0.02

@DATASETS.register_module()
class PybulletRandomDataset(GraspNet, CustomDataset):
    CLASSES = ('grasp', )

    def __init__(self,
                 root,
                 pipeline,
                 dump_folder=None,
                 rect_label_folder='rect_labels',
                 camera='kinect',
                 split='train',
                 view='1016',
                 test_mode=False):
        assert camera in ['kinect', 'realsense'], 'camera should be kinect or realsense'

        self.root = root
        self.camera = camera
        self.split = split
        self.collisionLabels = {}
        self.test_mode = test_mode
        self.dump_folder = dump_folder
        self.rect_label_folder = rect_label_folder

        if split == 'all':
            self.sceneIds = list(range(TOTAL_SCENE_NUM))
        elif split == 'train':
            self.sceneIds = list(range(500))
        elif split == 'train_0.6':
            self.sceneIds = list(range(300))
        elif split == 'train_0.3':
            self.sceneIds = list(range(150))



        self.data_infos = self.load_annotations()

        self.proposals = None


        if not test_mode:
            self._set_group_flag()

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self):
        data_infos = []
        for sceneId in self.sceneIds:
            for annId in range(20):
                rgb_filename = os.path.join("scenes", "scene_%05d" % sceneId, self.camera, "rgb", "%04d.png" % annId)
                # depth_filename = os.path.join("scene_%04d" % sceneId, self.camera, "depth", "%04d.png" % annId)
                depth_filename = os.path.join("scenes", "scene_%05d" % sceneId, self.camera, "depth", "%04d.png" % annId)
                origin_depth_filename = os.path.join("scenes", "scene_%05d" % sceneId, self.camera, "depth", "%04d.png" % annId)
                segment_filename = os.path.join("scenes", "scene_%05d" % sceneId, self.camera, "label", "%04d.png" % annId)
                img = Image.open(os.path.join(self.root, rgb_filename))
                width, height = img.size
                data_infos.append(
                    dict(sceneId=sceneId, annId=annId, camera=self.camera, rgb_filename=rgb_filename, depth_filename=depth_filename,
                         origin_depth_filename=origin_depth_filename, segment_filename=segment_filename,
                         width=width, height=height, domain='source')
                )

        return data_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['data_root'] = self.root
        results['image_fields'] = []
        results['rect_grasp_fields'] = []

    def get_ann_info(self, idx):
        sceneId = self.data_infos[idx]['sceneId']
        annId = self.data_infos[idx]['annId']
        rect_label_path = os.path.join(self.root, self.rect_label_folder, "scene_%05d" % sceneId, self.camera, "%04d.npy" % annId)
        # rect_label_path = os.path.join(self.root, "rect_labels", "scene_%04d" % sceneId, self.camera,
        #                                "%04d.npy" % annId)
        rect_labels = np.load(rect_label_path)
        center = rect_labels[:, :2]
        open = rect_labels[:, 2:4]
        height = rect_labels[:, 4]
        score = rect_labels[:, 5]
        object_id = rect_labels[:, 6]
        depth = rect_labels[:, 7]
        axis = open - center
        normal = np.concatenate((-axis[:, 1].reshape(-1, 1), axis[:, 0].reshape(-1, 1)), axis=1)
        normal = normal / np.linalg.norm(normal, axis=1).reshape(-1, 1) * height.reshape(-1, 1) / 2
        p1 = center + normal + axis
        p2 = center + normal - axis
        p3 = center - normal - axis
        p4 = center - normal + axis
        rect_grasps = np.concatenate((p1, p2, p3, p4), axis=1)
        depths = depth
        ann = dict(rect_grasps=rect_grasps,
                   depths=depths,
                   scores=score,
                   object_ids=object_id)
        return ann

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='grasp',
                 logger=None):
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)
        eval_results = OrderedDict()

        ge = MyGraspNetEval(root=self.root, camera=self.camera, split='test', view=self.view)
        dump_folder = os.path.join(os.path.abspath('.'), self.dump_folder)

        res, ap = ge.eval_seen(dump_folder, proc=4)
        # eval_results['AP_seen'] = np.mean(res)
        # res = res.transpose(3, 0, 1, 2).reshape(6, -1)
        # res = np.mean(res, axis=1)
        # eval_results['AP0.4_seen'] = res[1]
        # eval_results['AP0.8_seen'] = res[3]
        eval_results['AP_seen'] = ap[0]
        eval_results['AP0.4_seen'] = ap[1]
        eval_results['AP0.8_seen'] = ap[2]

        res, ap = ge.eval_similar(dump_folder, proc=4)
        # eval_results['AP_similar'] = np.mean(res)
        # res = res.transpose(3, 0, 1, 2).reshape(6, -1)
        # res = np.mean(res, axis=1)
        # eval_results['AP0.4_similar'] = res[1]
        # eval_results['AP0.8_similar'] = res[3]
        eval_results['AP_similar'] = ap[0]
        eval_results['AP0.4_similar'] = ap[1]
        eval_results['AP0.8_similar'] = ap[2]

        res, ap = ge.eval_novel(dump_folder, proc=4)
        eval_results['AP_novel'] = ap[0]
        eval_results['AP0.4_novel'] = ap[1]
        eval_results['AP0.8_novel'] = ap[2]
        log_msg = []
        log_msg.append(f'AP_seen: %f, AP0.4_seen: %f, AP0.8_seen: %f' % (eval_results['AP_seen'], eval_results['AP0.4_seen'], eval_results['AP0.8_seen']))
        log_msg.append(f'AP_similar: %f, AP0.4_similar: %f, AP0.8_similar: %f' % (
        eval_results['AP_similar'], eval_results['AP0.4_similar'], eval_results['AP0.8_similar']))
        log_msg.append(f'AP_novel %f, AP0.4_novel: %f, AP0.8_novel: %f' % (
            eval_results['AP_novel'], eval_results['AP0.4_novel'], eval_results['AP0.8_novel']))
        print_log(log_msg, logger=logger)
        return eval_results






