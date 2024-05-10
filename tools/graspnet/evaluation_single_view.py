from graspnet.my_graspnet_eval import MyGraspNetEval
import os



graspnet_root = r'/home/qrrr/robotic-grasping/planer_graspnet'
ge = MyGraspNetEval(root=graspnet_root, camera='kinect', split='test', view='1016')

dump_folder = os.path.join('/home/qrrr/robotic-grasping/logs/220912_2135_kinect/epoch_20_output')
# dump_folder = os.path.join('/home/qrrr/ggcnn/output/models/220912_2028_kinect_D/epoch_40_output')


res, ap = ge.eval_seen(dump_folder, proc=4)
print("AP_seen: %f" % ap[0])
print("AP0.4_seen: %f" % ap[1])
print("AP0.8_seen: %f" % ap[2])
res, ap = ge.eval_similar(dump_folder, proc=4)
print("AP_similar: %f" % ap[0])
print("AP0.4_similar: %f" % ap[1])
print("AP0.8_similar: %f" % ap[2])
res, ap = ge.eval_novel(dump_folder, proc=4)
print("AP_novel: %f" % ap[0])
print("AP0.4_novel: %f" % ap[1])
print("AP0.8_novel: %f" % ap[2])

