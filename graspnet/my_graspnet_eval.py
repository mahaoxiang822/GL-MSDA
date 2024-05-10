from graspnetAPI import GraspNetEval, GraspGroup
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import get_scene_name, create_table_points, parse_posevector, load_dexnet_model, transform_points, compute_point_distance, compute_closest_points, voxel_sample_points, topk_grasps, get_grasp_score, collision_detection, eval_grasp
import numpy as np
import os
import open3d as o3d
from graspnetAPI.utils.utils import generate_scene_model
from .my_grasp import MyRectGraspGroup
from tqdm import tqdm


class MyGraspNetEval(GraspNetEval):
    def __init__(self, root, camera, split='test', view='1016'):
        super(MyGraspNetEval, self).__init__(root, camera, split)
        self.view = view

    def eval_scene(self, scene_id, dump_folder, TOP_K=50, return_list=False, vis=False, max_width=0.1):
        '''
        **Input:**

        - scene_id: int of the scene index.

        - dump_folder: string of the folder that saves the dumped npy files.

        - TOP_K: int of the top number of grasp to evaluate

        - return_list: bool of whether to return the result list.

        - vis: bool of whether to show the result

        - max_width: float of the maximum gripper width in evaluation

        **Output:**

        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

        list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

        scene_dir = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'rgb')
        with open(os.path.join(self.root, 'views', get_scene_name(scene_id), self.camera, 'scene_' + self.view + '.txt'), 'r') as f:
            annIds = f.readlines()

        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=int(annIds[0].split('.')[0]))

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []
        for i in tqdm(range(len(annIds)), 'Scene:{}'.format(scene_id)):
            ann_id = int(annIds[i])

            # 如果保存的是rect_labels，先转换
            # rect_labels = MyRectGraspGroup(
            #     os.path.join(dump_folder, get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,)))
            # grasp_group = rect_labels.to_grasp_group(self.camera, depths=None, depth_method=None)
            # 否则，直接加载
            grasp_group = GraspGroup().from_npy(
                os.path.join(dump_folder, get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,)))

            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # clip width to [0,max_width]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = (gg_array[:, 1] < 0)
            max_width_mask = (gg_array[:, 1] > max_width)
            gg_array[min_width_mask, 1] = 0
            gg_array[max_width_mask, 1] = max_width
            grasp_group.grasp_group_array = gg_array

            grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list,
                                                                     pose_list, config, table=table_trans,
                                                                     voxel_size=0.008, TOP_K=TOP_K)

            # remove empty
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                print('\rMean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id), np.mean(grasp_accuracy[:, :]),
                      end='')
                continue

            # concat into scene level
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(
                score_list), np.concatenate(collision_mask_list)

            if vis:
                t = o3d.geometry.PointCloud()
                t.points = o3d.utility.Vector3dVector(table_trans)
                model_list = generate_scene_model(self.root, 'scene_%04d' % scene_id, ann_id, return_poses=False,
                                                  align=False, camera=self.camera)
                import copy
                gg = GraspGroup(copy.deepcopy(grasp_list))
                scores = np.array(score_list)
                scores = scores / 2 + 0.5  # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0.3
                gg.scores = scores
                gg.widths = 0.1 * np.ones((len(gg)), dtype=np.float32)
                grasps_geometry = gg.to_open3d_geometry_list()
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                o3d.visualization.draw_geometries([pcd, *grasps_geometry])
                o3d.visualization.draw_geometries([pcd, *grasps_geometry, *model_list])
                o3d.visualization.draw_geometries([*grasps_geometry, *model_list, t])

            # sort in scene level
            grasp_confidence = grasp_list[:, 0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[
                indices]

            no_collision_num = np.sum(~collision_mask_list)

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            # calculate AP
            grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0, TOP_K):
                    if k + 1 > len(score_list):
                        grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (
                                    k + 1)
                    else:
                        grasp_accuracy[k, fric_idx] = np.sum(
                            ((score_list[0:k + 1] <= fric) & (score_list[0:k + 1] > 0)).astype(int)) / (k + 1)

            # print('\rMean Accuracy for scene:%04d ann:%04d = %.3f' % (
            # scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:, :])), end='', flush=True)
            scene_accuracy.append(grasp_accuracy)
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list

    def eval_train(self, dump_folder, proc = 2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for all split.
        '''
        res = self.parallel_eval_scenes(scene_ids=list(range(0, 1)), dump_folder=dump_folder, proc=proc)
        # ap = np.mean(res)
        # print('\nEvaluation Result:\n----------\n{}, AP Seen={}'.format(self.camera, ap))
        ap = 0
        ap_4 = 0
        ap_8 = 0
        num = 0
        topk = 50
        num_coe_of_friction = 6
        for i in range(len(res)):
            ap += np.sum(np.array(res[i]))
            ap_4 += np.sum(np.array(res[i])[:, :, 1])
            ap_8 += np.sum(np.array(res[i])[:, :, 3])
            num += len(res[i])
        ap = ap / (num * topk * num_coe_of_friction)
        ap_4 = ap_4 / (num * topk)
        ap_8 = ap_8 / (num * topk)
        # print('\nEvaluation Result:\n----------\n{}, AP={}, AP Novel={}'.format(self.camera, ap, ap))
        return res, [ap, ap_4, ap_8]

    def eval_seen(self, dump_folder, proc = 2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for seen split.
        '''
        res = self.parallel_eval_scenes(scene_ids = list(range(100, 130)), dump_folder = dump_folder, proc = proc)
        # ap = np.mean(res)
        # print('\nEvaluation Result:\n----------\n{}, AP Seen={}'.format(self.camera, ap))
        ap = 0
        ap_4 = 0
        ap_8 = 0
        num = 0
        topk = 50
        num_coe_of_friction = 6
        for i in range(len(res)):
            ap += np.sum(np.array(res[i]))
            ap_4 += np.sum(np.array(res[i])[:, :, 1])
            ap_8 += np.sum(np.array(res[i])[:, :, 3])
            num += len(res[i])
        ap = ap / (num * topk * num_coe_of_friction)
        ap_4 = ap_4 / (num * topk)
        ap_8 = ap_8 / (num * topk)
        # print('\nEvaluation Result:\n----------\n{}, AP={}, AP Novel={}'.format(self.camera, ap, ap))
        return res, [ap, ap_4, ap_8]

    def eval_similar(self, dump_folder, proc = 2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for similar split.
        '''
        res = self.parallel_eval_scenes(scene_ids = list(range(130, 160)), dump_folder = dump_folder, proc = proc)
        # ap = np.mean(res)
        # print('\nEvaluation Result:\n----------\n{}, AP={}, AP Similar={}'.format(self.camera, ap, ap))
        ap = 0
        ap_4 = 0
        ap_8 = 0
        num = 0
        topk = 50
        num_coe_of_friction = 6
        for i in range(len(res)):
            ap += np.sum(np.array(res[i]))
            ap_4 += np.sum(np.array(res[i])[:, :, 1])
            ap_8 += np.sum(np.array(res[i])[:, :, 3])
            num += len(res[i])
        ap = ap / (num * topk * num_coe_of_friction)
        ap_4 = ap_4 / (num * topk)
        ap_8 = ap_8 / (num * topk)
        # print('\nEvaluation Result:\n----------\n{}, AP={}, AP Novel={}'.format(self.camera, ap, ap))
        return res, [ap, ap_4, ap_8]

    def eval_novel(self, dump_folder, proc=2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for novel split.
        '''
        res = self.parallel_eval_scenes(scene_ids=list(range(160, 190)), dump_folder=dump_folder, proc=proc)
        ap = 0
        ap_4 = 0
        ap_8 = 0
        num = 0
        topk = 50
        num_coe_of_friction = 6
        for i in range(len(res)):
            ap += np.sum(np.array(res[i]))
            ap_4 += np.sum(np.array(res[i])[:, :, 1])
            ap_8 += np.sum(np.array(res[i])[:, :, 3])
            num += len(res[i])
        ap = ap / (num * topk * num_coe_of_friction)
        ap_4 = ap_4 / (num * topk)
        ap_8 = ap_8 / (num * topk)
        # print('\nEvaluation Result:\n----------\n{}, AP={}, AP Novel={}'.format(self.camera, ap, ap))
        return res, [ap, ap_4, ap_8]