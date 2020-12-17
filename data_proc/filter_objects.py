#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: willc
"""

import numpy as np
from utils import py_cpu_nms
import cv2

file_name = "/Volumes/Will 1/thesis/cus_ktti_vis/data_proc/data/000015_pred.txt"

class Object3d(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        self.data = label_file_line.split(" ")
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.score = 0.0
        if len(data) > 15:
            self.score =  data[15]
        # print(self.score)


    def estimate_diffculty(self):
        """ Function that estimate difficulty to detect the object as defined in kitti website"""
        # height of the bounding box
        bb_height = np.abs(self.xmax - self.xmin)

        if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
            return "Easy"
        elif bb_height >= 25 and self.occlusion in [0, 1] and self.truncation <= 0.30:
            return "Moderate"
        elif (
            bb_height >= 25 and self.occlusion in [0, 1, 2] and self.truncation <= 0.50
        ):
            return "Hard"
        else:
            return "Unknown"

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty of estimation: {}".format(self.estimate_diffculty()))

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def split_predict_result(objects):
    # labels, lists, prefixs(start from zero)
    labels = []
    objs = []
    prefix = []
    for idx, obj in enumerate(objects):
        label_name = obj.type
        if label_name not in labels:
            labels.append(label_name)
            prefix.append(idx)
            objs.append([])

        x0, y0, x1, y1, score = obj.xmin, obj.ymin, obj.xmax, obj.ymax, (obj.ymax-obj.ymin)*(obj.xmax-obj.xmin)
        objs[-1].append([x0, y0, x1, y1, score])
    return labels, np.asarray(objs), prefix


def objects_to_label(objects, output_name):
    with open(output_name, "w") as text_file:
        for obj in objects:
            text = " ".join(obj.data)
            text_file.write(text + "\n")
    print("successfully")



def nms_box():
    pass

def nms_3d():
    pass

if __name__ == "__main__":
    objects = read_label(file_name)
    labels, objs_list, prefix = split_predict_result(objects)
    # print(labels)
    # print(objs_list)
    # print(prefix)

    image = cv2.imread("/Volumes/Will 1/thesis/cus_ktti_vis/data_proc/data/000015.png")

    new_objects = []
    for idx, objs in enumerate(objs_list):
        # print(objs)
        for obj in objs:
            cv2.rectangle(image, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])), (0, 255, 0), 2)
        keep = (py_cpu_nms(np.asarray(objs), 0.1))
        for k in keep:
            new_objects.append(objects[k + prefix[idx]])

    print((new_objects))
    objects_to_label(new_objects, "000015_pred_mod.txt")

    cv2.imwrite("./test.jpg", image)