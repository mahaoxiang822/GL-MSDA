import os
import numpy as np
import cv2
from xml.dom.minidom import Document, parse
import xml.dom.minidom

def load_annotations(path):
    rect_grasps = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            x1, y1 = lines[i].split(' ')[0:2]
            x2, y2 = lines[i + 1].split(' ')[0:2]
            x3, y3 = lines[i + 2].split(' ')[0:2]
            x4, y4 = lines[i + 3].split(' ')[0:2]
            rect_grasps.append([float(x1), float(y1), float(x2), float(y2), float(x3),
                                float(y3), float(x4), float(y4)])
    return np.array(rect_grasps)


def write_xml_files(filename, path, grasp_list, w, h, d):
    id = filename[0:7]
    # dict_box[filename]=json_dict[filename]
    doc = xml.dom.minidom.Document()  # 创建DOM树对象
    root = doc.createElement('annotation')  # 创建根结点，每次都要用DOM对象来创建任何结点
    doc.appendChild(root)  # 用dom对象添加根节点

    foldername = doc.createElement("folder")  # 用DOM对象创建元素节点
    foldername.appendChild(doc.createTextNode("Cornell"))  # 添加子节点
    root.appendChild(foldername)

    nodeFilename = doc.createElement('filename')
    nodeFilename.appendChild(doc.createTextNode(filename))
    root.appendChild(nodeFilename)

    sourcename = doc.createElement("source")

    databasename = doc.createElement("database")
    databasename.appendChild(doc.createTextNode("The Cornell Database"))
    sourcename.appendChild(databasename)

    annotationname = doc.createElement("annotation")
    annotationname.appendChild(doc.createTextNode("Cornell"))
    sourcename.appendChild(annotationname)

    imagename = doc.createElement("image")
    imagename.appendChild(doc.createTextNode(id + "r.png"))
    sourcename.appendChild(imagename)

    root.appendChild(sourcename)


    nodesize = doc.createElement('size')
    nodewidth = doc.createElement('width')
    nodewidth.appendChild(doc.createTextNode(str(w)))
    nodesize.appendChild(nodewidth)
    nodeheight = doc.createElement('height')
    nodeheight.appendChild(doc.createTextNode(str(h)))
    nodesize.appendChild(nodeheight)
    nodedepth = doc.createElement('depth')
    nodedepth.appendChild(doc.createTextNode(str(d)))
    nodesize.appendChild(nodedepth)
    root.appendChild(nodesize)

    segname = doc.createElement("segmented")
    segname.appendChild(doc.createTextNode("0"))
    root.appendChild(segname)

    for grasp in grasp_list:
        nodeobject = doc.createElement('object')
        nodename = doc.createElement('name')
        nodename.appendChild(doc.createTextNode(str('grasp')))
        nodeobject.appendChild(nodename)
        nodebndbox = doc.createElement('rect_grasp')
        nodex1 = doc.createElement('x1')
        nodex1.appendChild(doc.createTextNode(str(grasp[0])))
        nodebndbox.appendChild(nodex1)
        nodey1 = doc.createElement('y1')
        nodey1.appendChild(doc.createTextNode(str(grasp[1])))
        nodebndbox.appendChild(nodey1)
        nodex2 = doc.createElement('x2')
        nodex2.appendChild(doc.createTextNode(str(grasp[2])))
        nodebndbox.appendChild(nodex2)
        nodey2 = doc.createElement('y2')
        nodey2.appendChild(doc.createTextNode(str(grasp[3])))
        nodebndbox.appendChild(nodey2)
        nodex3 = doc.createElement('x3')
        nodex3.appendChild(doc.createTextNode(str(grasp[4])))
        nodebndbox.appendChild(nodex3)
        nodey3 = doc.createElement('y3')
        nodey3.appendChild(doc.createTextNode(str(grasp[5])))
        nodebndbox.appendChild(nodey3)
        nodex4 = doc.createElement('x4')
        nodex4.appendChild(doc.createTextNode(str(grasp[6])))
        nodebndbox.appendChild(nodex4)
        nodey4 = doc.createElement('y4')
        nodey4.appendChild(doc.createTextNode(str(grasp[7])))
        nodebndbox.appendChild(nodey4)

        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)
    fp = open(os.path.join(path, filename), 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


if __name__ == '__main__':
    root_path = '/home/qinran_2020/data/cornell'
    img_path = os.path.join(root_path, 'rgb')
    xml_path = os.path.join(root_path, 'Annotations')
    txt_path = os.path.join(root_path, 'labels')
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)
    txts = os.listdir(txt_path)
    for t in txts:
        print(t)
        grasps = load_annotations(os.path.join(txt_path, t))
        id = t[0:7]
        xml_name = id + 'cpos.xml'
        img_name = os.path.join(img_path, id + 'r.png')
        img = cv2.imread(img_name)
        h, w, d = img.shape
        write_xml_files(xml_name, xml_path, grasps, w, h, d)


