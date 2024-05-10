from graspnetAPI import RectGraspGroup, RectGrasp, GraspGroup
import copy
import numpy as np
import cv2
from graspnetAPI.utils.utils import (batch_framexy_depth_2_xyz, batch_key_point_2_rotation,
                                     batch_center_depth, get_camera_intrinsic)

GRASP_ARRAY_LEN = 17
RECT_GRASP_ARRAY_LEN = 8
EPS = 1e-8


class MyRectGrasp(RectGrasp):
    def __init__(self, *args):
        '''
        **Input:**

        - args can be a numpy array or tuple of the center_x, center_y, open_x, open_y, height, score, object_id, depth

        - the format of numpy array is [center_x, center_y, open_x, open_y, height, score, object_id, depth]

        - the length of the numpy array is 8.
        '''
        if len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.rect_grasp_array = copy.deepcopy(args[0])
            else:
                raise TypeError('if only one arg is given, it must be np.ndarray.')
        elif len(args) == RECT_GRASP_ARRAY_LEN:
            self.rect_grasp_array = np.array(args).astype(np.float64)
        else:
            raise ValueError('only one or six arguments are accepted')

    def __repr__(self):
        return 'Rectangle Grasp: score:{}, height:{}, open point:{}, center point:{}, depth:{}, object id:{}'.format(self.score, self.height, self.open_point, self.center_point, self.depth, self.object_id)

    @property
    def depth(self):
        '''
        **Output:**

        - float of the depth.
        '''
        return self.rect_grasp_array[7]

    @depth.setter
    def depth(self, depth):
        '''
        **input:**

        - float of the depth.
        '''
        self.rect_grasp_array[7] = depth



class MyRectGraspGroup(RectGraspGroup):
    def __init__(self, *args):
        '''
        **Input:**

        - args can be (1) nothing (2) numpy array of rect_grasp_group_array (3) str of the numpy file.
        '''
        if len(args) == 0:
            self.rect_grasp_group_array = np.zeros((0, RECT_GRASP_ARRAY_LEN), dtype=np.float64)
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.rect_grasp_group_array = args[0]
            elif isinstance(args[0], str):
                self.rect_grasp_group_array = np.load(args[0])
            else:
                raise ValueError('args must be nothing, numpy array or string.')
        else:
            raise ValueError('args must be nothing, numpy array or string.')

    def __repr__(self):
        repr = '----------\nRectangle Grasp Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 10:
            for rect_grasp_array in self.rect_grasp_group_array:
                repr += MyRectGrasp(rect_grasp_array).__repr__() + '\n'
        else:
            for i in range(5):
                repr += MyRectGrasp(self.rect_grasp_group_array[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(5):
                repr += MyRectGrasp(self.rect_grasp_group_array[-(5-i)]).__repr__() + '\n'
        return repr + '----------'

    @property
    def depths(self):
        '''
        **Output:**

        - float of the depth.
        '''
        return self.rect_grasp_group_array[:, 7]

    @depths.setter
    def depths(self, depths):
        '''
        **input:**

        - float of the depth.
        '''
        assert depths.size == len(self)
        self.rect_grasp_group_array[:, 7] = depths

    def batch_get_key_points(self):
        '''
        **Output:**

        - center, open_point, upper_point, each of them is a numpy array of shape (2,)
        '''
        open_points = self.open_points # (-1, 2)
        centers = self.center_points # (-1, 2)
        heights = (self.heights).reshape((-1, 1)) # (-1, )
        open_point_vector = open_points - centers
        norm_open_point_vector = np.linalg.norm(open_point_vector, axis = 1).reshape(-1, 1)
        unit_open_point_vector = open_point_vector / np.hstack((norm_open_point_vector, norm_open_point_vector)) # (-1, 2)
        counter_clock_wise_rotation_matrix = np.array([[0,-1], [1, 0]])
        # upper_points = np.dot(counter_clock_wise_rotation_matrix, unit_open_point_vector.reshape(-1, 2, 1)).reshape(-1, 2) * np.hstack([heights, heights]) / 2 + centers # (-1, 2)
        upper_points = np.einsum('ij,njk->nik', counter_clock_wise_rotation_matrix, unit_open_point_vector.reshape(-1, 2, 1)).reshape(-1, 2) * np.hstack([heights, heights]) / 2 + centers # (-1, 2)
        return centers, open_points, upper_points


    def to_grasp_group(self, camera, depths, depth_method=batch_center_depth, img_shape=None):
        '''
        **Input:**

        - camera: string of type of camera, 'kinect' or 'realsense'.

        - depths: numpy array of the depths image.

        - depth_method: function of calculating the depth.

        **Output:**

        - grasp_group: GraspGroup instance or None.

        .. note:: The number may not be the same to the input as some depth may be invalid.
        '''
        centers, open_points, upper_points = self.batch_get_key_points()
        # print(f'centers:{centers}\nopen points:{open_points}\nupper points:{upper_points}')
        depth_default = 0.02
        if depth_method == None:
            depths_2d = self.depths / 1000.0 - depth_default
        else:
            depths_2d = depth_method(depths.squeeze(), centers, open_points, upper_points) / 1000.0
        # print(f'depths_3d:{depths_2d}')
        valid_mask1 = np.abs(depths_2d) > EPS
        valid_mask2 = np.linalg.norm(centers - open_points, axis =1) > EPS
        valid_mask3 = np.linalg.norm(centers - upper_points, axis =1) > EPS
        valid_mask4 = np.linalg.norm(upper_points - open_points, axis =1) > EPS
        valid_mask = np.logical_and(
            np.logical_and(valid_mask1, valid_mask2),
            np.logical_and(valid_mask3, valid_mask4)
        )
        # print(f'valid_mask:{valid_mask}')
        centers = centers[valid_mask]
        open_points = open_points[valid_mask]
        upper_points = upper_points[valid_mask]
        # print(f'## After filtering\ncenters:{centers}\nopen points:{open_points}\nupper points:{upper_points}')
        depths_2d = depths_2d[valid_mask]
        valid_num = centers.shape[0]
        if valid_num == 0:
            return None
        centers_xyz = np.array(batch_framexy_depth_2_xyz(centers[:, 0], centers[:, 1], depths_2d, camera)).T
        open_points_xyz = np.array(batch_framexy_depth_2_xyz(open_points[:, 0], open_points[:, 1], depths_2d, camera)).T
        upper_points_xyz = np.array(batch_framexy_depth_2_xyz(upper_points[:, 0], upper_points[:, 1], depths_2d, camera)).T
        depths = depth_default * np.ones((valid_num, 1))
        heights = (np.linalg.norm(upper_points_xyz - centers_xyz, axis = 1) * 2).reshape((-1, 1))
        widths = (np.linalg.norm(open_points_xyz - centers_xyz, axis = 1) * 2).reshape((-1, 1))
        scores = (self.scores)[valid_mask].reshape((-1, 1))
        object_ids = (self.object_ids)[valid_mask].reshape((-1, 1))
        translations = centers_xyz
        rotations = batch_key_point_2_rotation(centers_xyz, open_points_xyz, upper_points_xyz).reshape((-1, 9))
        grasp_group = GraspGroup()
        grasp_group.grasp_group_array = copy.deepcopy(np.hstack((scores, widths, heights, depths, rotations, translations, object_ids))).astype(np.float64)
        return grasp_group

    def to_opencv_image(self, opencv_rgb, numGrasp=0):
        '''
        **input:**

        - opencv_rgb: numpy array of opencv BGR format.

        - numGrasp: int of the number of grasp, 0 for all.

        **Output:**

        - numpy array of opencv RGB format that shows the rectangle grasps.
        '''
        img = copy.deepcopy(opencv_rgb)
        if numGrasp == 0:
            numGrasp = self.__len__()
        shuffled_rect_grasp_group_array = copy.deepcopy(self.rect_grasp_group_array)
        np.random.shuffle(shuffled_rect_grasp_group_array)
        for rect_grasp_array in shuffled_rect_grasp_group_array[:numGrasp]:
            center_x, center_y, open_x, open_y, height, score, object_id = rect_grasp_array
            center = np.array([center_x, center_y])
            left = np.array([open_x, open_y])
            axis = left - center
            normal = np.array([-axis[1], axis[0]])
            normal = normal / np.linalg.norm(normal) * height / 2
            p1 = center + normal + axis
            p2 = center + normal - axis
            p3 = center - normal - axis
            p4 = center - normal + axis
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2, 8)
            cv2.line(img, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (255, 0, 0), 4, 8)
            cv2.line(img, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (0, 0, 255), 2, 8)
            cv2.line(img, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), (255, 0, 0), 4, 8)
        return img
