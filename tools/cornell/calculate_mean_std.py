import os
import cv2
import numpy as np

data_root = '/home/qinran_2020/data/cornell/rgb'
R = 0
G = 0
B = 0
R_2 = 0
G_2 = 0
B_2 = 0
N = 0

filelist = os.listdir(data_root)

for i in filelist:
    annId = int(i[3:7])
    filename = os.path.join(data_root, "pcd%04dr.png" % annId)
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    h, w, c = img.shape
    N += h * w

    R += np.sum(img[:, :, 0])
    R_2 += np.sum(np.power(img[:, :, 0], 2.0))
    G += np.sum(img[:, :, 1])
    G_2 += np.sum(np.power(img[:, :, 1], 2.0))
    B += np.sum(img[:, :, 2])
    B_2 += np.sum(np.power(img[:, :, 2], 2.0))

R_mean = R/N
G_mean = G/N
B_mean = B/N

R_std = np.sqrt(R_2/N - R_mean * R_mean)
G_std = np.sqrt(G_2/N - G_mean * G_mean)
B_std = np.sqrt(B_2/N - B_mean * B_mean)

print("R_mean: %f, G_mean: %f, B_mean: %f" % (R_mean, G_mean, B_mean))
print("R_std: %f, G_std: %f, B_std: %f" % (R_std, G_std, B_std))