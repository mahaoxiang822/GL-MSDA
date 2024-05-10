import os

graspnet_root = r'/home/qrrr/mmdetection_grasp/data/planer_graspnet'
camera = 'kinect'
train_nums = 0
test_nums = 0
for i in range(100):
    with open(os.path.join(graspnet_root, "views", "scene_%04d" % i, camera, "scene_1016.txt"), "r") as f:
        view_list = f.readlines()
    train_nums += len(view_list)
for i in range(100, 190):
    with open(os.path.join(graspnet_root, "views", "scene_%04d" % i, camera, "scene_1016.txt"), "r") as f:
        view_list = f.readlines()
    test_nums += len(view_list)
print(camera + "_train_nums: " + str(train_nums))
print(camera + "_test_nums: " + str(test_nums))