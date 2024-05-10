from graspnet.my_grasp import MyRectGraspGroup
import os
import cv2
import open3d as o3d
from collections import Counter
import numpy as np
from graspnetAPI import GraspGroup, RectGraspGroup, GraspNet


def grasp_center_refine(grasp_group, pcd):
    refined_gg = []
    for grasp in grasp_group:
        rot = grasp.rotation_matrix
        center = grasp.translation
        depth = grasp.depth
        width = min(grasp.width, 0.095)
        approach = rot[:, 0]
        approach_norm = approach / (np.sum(approach ** 2) ** 0.5)
        obbox = o3d.geometry.OrientedBoundingBox(center=center + (-0.02 + (depth + 0.02) / 2) * approach_norm, R=rot,
                                                 extent=np.array([depth + 0.02, width, 0.02]))
        # o3d_pcd_copy = copy.deepcopy(o3d_pcd)
        crop_points = pcd.crop(obbox)
        new_center = crop_points.get_center()
        _, _, z = grasp.translation
        new_center[2] = z
        # new_depth = 0.04
        # move the center to the eelink frame
        # new_center = new_center - rot[:, 0] * (np.vstack((new_depth-depth, new_depth-depth, new_depth-depth)).T)
        grasp.translation = new_center
        # grasp.depth = 0.04
        grasp.width = width
        refined_gg.append(grasp.grasp_array)
    refined_gg = GraspGroup(np.asarray(refined_gg))
    return refined_gg

def vis(graspnet_root, sceneId, annId, camera, gg_path):
    g = GraspNet(graspnet_root, camera=camera, split='test')
    bgr = g.loadBGR(sceneId=sceneId, camera=camera, annId=annId)
    depth = g.loadDepth(sceneId=sceneId, camera=camera, annId=annId)

    grasp_group_array = np.load(gg_path)
    grasp_group = GraspGroup()
    grasp_group.grasp_group_array = grasp_group_array

    grasp_group = grasp_group.nms(0.04, 45.0/180*np.pi)
    # grasp_score = grasp_group.scores
    # index = sorted(range(len(grasp_score)), key=lambda x:-grasp_score[x])
    # grasp_group.grasp_group_array = grasp_group.grasp_group_array[index][:10]
    # mask = grasp_score > 0.5
    # grasp_group.grasp_group_array = grasp_group.grasp_group_array[mask]
    # grasp_group = grasp_group.random_sample(20)
    pcd = g.loadScenePointCloud(sceneId, camera, annId, align=False)

    # grasp_group = grasp_center_refine(grasp_group, pcd)

    rect_grasp = grasp_group.to_rect_grasp_group(camera)
    rect_grasp_new = MyRectGraspGroup()
    rect_grasp_new.rect_grasp_group_array = rect_grasp.rect_grasp_group_array
    img = rect_grasp_new.to_opencv_image(bgr)
    # cv2.imwrite(os.path.join("images", "rect_realsense_" + str(sceneId) + ".png"), img)

    geometry = []
    geometry.append(pcd)
    geometry += grasp_group.to_open3d_geometry_list()
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # geometry.append(frame)
    # o3d.visualization.draw_geometries(geometry)
    return geometry, img