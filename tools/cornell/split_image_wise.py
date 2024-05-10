import os

def write_txt(path, ids):
    with open(path, "w") as f:
        for i in range(len(ids)):
            f.write("%04d\n" % ids[i])


root_path = '/home/qinran_2020/data/cornell'
img_path = os.path.join(root_path, 'rgb')
imgs = sorted(os.listdir(img_path))
img_set_path = os.path.join(root_path, 'ImageSets', 'Main', 'iw')
if not os.path.exists(img_set_path):
    os.makedirs(img_set_path)
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []


for i in range(len(imgs)):
    id = int(imgs[i][3:7])
    if i % 5 == 0:
        data1.append(id)
    elif i % 5 == 1:
        data2.append(id)
    elif i % 5 == 2:
        data3.append(id)
    elif i % 5 == 3:
        data4.append(id)
    else:
        data5.append(id)
write_txt(os.path.join(img_set_path, "train1.txt"), sorted(data1 + data2 + data3 + data4))
write_txt(os.path.join(img_set_path, "test1.txt"), sorted(data5))
write_txt(os.path.join(img_set_path, "train2.txt"), sorted(data1 + data2 + data3 + data5))
write_txt(os.path.join(img_set_path, "test2.txt"), sorted(data4))
write_txt(os.path.join(img_set_path, "train3.txt"), sorted(data1 + data2 + data4 + data5))
write_txt(os.path.join(img_set_path, "test3.txt"), sorted(data3))
write_txt(os.path.join(img_set_path, "train4.txt"), sorted(data1 + data3 + data4 + data5))
write_txt(os.path.join(img_set_path, "test4.txt"), sorted(data2))
write_txt(os.path.join(img_set_path, "train5.txt"), sorted(data2 + data3 + data4 + data5))
write_txt(os.path.join(img_set_path, "test5.txt"), sorted(data1))