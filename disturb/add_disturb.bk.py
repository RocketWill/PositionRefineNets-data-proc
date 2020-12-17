#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: willc
"""

import os
import cv2
import numpy as np
import open3d as o3d
from utils import read_calib_file, load_label, load_velo_scan, project_cam2_to_velo, project_velo_to_cam2, project_camera_to_lidar \
    , project_to_image
from random import random, shuffle

def view_cloud(source, target):
    source.paint_uniform_color([0, 1, 0])
    target.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([source, target])

def get_random_num(rang):
    min_, max_ = rang
    value = random()
    scaled_value = min_ + (value * (max_ - min_))
    return round(scaled_value, 3)

def random_split_pc(selected_pc):
    start_range = [0, 1, 2]
    pc_len = len(selected_pc)
    mid = pc_len//2
    rn = int(get_random_num((pc_len/4.0, pc_len/2.)))
    shuffle(start_range)
    start = start_range[0]

    if start == 0:
        mod = selected_pc[:rn]
        keep = selected_pc[rn:]
    elif start == 2:
        keep = selected_pc[:rn]
        mod = selected_pc[rn:]
    else:
        s = mid - rn // 2
        keep = np.concatenate((selected_pc[: s], selected_pc[s+rn:]), axis=0)
        # keep = selected_pc[: s] + selected_pc[s+rn:]
        mod = selected_pc[s: s+rn]
    return keep, mod


def get_randon_pc(selected_pc, forward_range=(0.2, 0.5), lr_range=(-0.3, 0.3)):
    fr = get_random_num(forward_range)
    lrr = get_random_num(lr_range)
    keep, mod = random_split_pc(selected_pc)
    mod[:, 0] = mod[:, 0] - fr
    mod[:, 1] = mod[:, 1] - lrr
    return np.concatenate((keep, mod), axis=0)

def get_obj_range_info(obj):
    proj_cam_to_velo = project_cam2_to_velo(calib)
    x, y, z = obj.x, obj.y, obj.z
    h, w, l = obj.h, obj.w, obj.l
    points = [
        [x, y, z],
        [x, y-h, z],
        [x+w/2, y, z],
        [x-w/2, y, z],
        [x, y, z+l/2],
        [x, y, z - l / 2]
    ]
    points = np.asarray(points)
    pts_velo = project_camera_to_lidar(points.transpose(), proj_cam_to_velo)
    return pts_velo.transpose()

if __name__ == '__main__':
    calib_path = "/Users/willc/Desktop/cus_ktti_vis/000073_calib.txt"
    label_path = "/Users/willc/Desktop/cus_ktti_vis/000073_label.txt"
    velo_path = "/Users/willc/Desktop/cus_ktti_vis/000073.bin"

    # Pedestrian 0.00 1 0.80 189.46 158.23 256.19 356.44 1.70 0.61  0.31
    # Load calibration
    calib = read_calib_file(calib_path)

    # Load labels
    labels = load_label(label_path)

    # Load Lidar PC
    pc_velo = load_velo_scan(velo_path)[:, :3]

    proj_cam_to_velo = project_cam2_to_velo(calib)

    points = [[2.28, 1.63, 10.51], [2.28, 1.63-1.75, 10.51], \
              [2.28+0.63/2, 1.63, 10.51], [2.28-0.63/2, 1.63, 10.51], \
              [2.28, 1.63, 10.51+0.51/2],[2.28, 1.63, 10.51-0.51/2]]


    points = np.asarray(points)
    print(points.transpose().shape)

    pts_velo = project_camera_to_lidar(points.transpose(), proj_cam_to_velo)
    print(pts_velo.transpose())

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points =o3d.utility.Vector3dVector(pc_velo)


    pc_velo = np.asarray(pc_velo)
    inds = np.where(\
                    # (pc_velo[:, 1] >= -2.10622237) \
                    # & (pc_velo[:, 1] <= -2.09733604) \
                    # (pc_velo[:, 0] >= 10.82672545) \
                    # & (pc_velo[:, 0] <= 10.83768626) \
                    # & (pc_velo[:, 2] >= -1.62085398) \
                    # & (pc_velo[:, 2] <= 0.13095698) \
                    (pc_velo[:, 2] <= 0.13095698) \
                    & (pc_velo[:, 2] >= -1.61898463) \
                    & (pc_velo[:, 1] <= -1.78682891)
                    & (pc_velo[:, 1] >= -2.4167295)
                    & (pc_velo[:, 0] <= 11.08716028)
                    & (pc_velo[:, 0] >= 10.57725143)
                    )
    print(inds)
    print(pc_velo)
    selected = pc_velo[inds]
    selected = selected[selected[:, 2].argsort()]
    # print(selected)
    # selected[:, 0] = selected[:, 0] - 0.3 # negative forward
    selected = get_randon_pc(selected)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(selected)
    view_cloud(point_cloud, pcd)

