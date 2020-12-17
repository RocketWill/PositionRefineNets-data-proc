#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: willc
"""

import os
import struct
import numpy as np
import open3d as o3d
import glob
import ntpath
import argparse
import tqdm
import matplotlib.pyplot as plt
import cv2
from utils import read_calib_file, load_label, load_velo_scan, project_cam2_to_velo, \
    project_velo_to_cam2, project_camera_to_lidar, project_to_image
from random import random, shuffle

def makedirs(path):
    os.makedirs(path, exist_ok=True)
    print("Make new dir: {}".format(path))

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


def get_randon_pc(selected_pc, forward_range=(0.2, 0.65), lr_range=(-0.45, 0.45)):
    fr = get_random_num(forward_range)
    lrr = get_random_num(lr_range)
    keep, mod = random_split_pc(selected_pc)
    mod[:, 0] = mod[:, 0] - fr
    mod[:, 1] = mod[:, 1] - lrr
    return np.concatenate((keep, mod), axis=0)

def get_obj_range_info(obj):
    proj_cam_to_velo = project_cam2_to_velo(calib)
    x, y, z = obj.t[0], obj.t[1], obj.t[2]
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

def get_obj_inds(pc_velo, range_info):
    z1, z2 = range_info[0][2], range_info[1][2]
    y1, y2 = range_info[2][1], range_info[3][1]
    x1, x2 = range_info[4][0], range_info[5][0]
    inds = np.where(
        (pc_velo[:, 2] <= max(z1, z2))
        & (pc_velo[:, 2] >= min(z1, z2))
        & (pc_velo[:, 1] <= max(y1, y2))
        & (pc_velo[:, 1] >= min(y1, y2))
        & (pc_velo[:, 0] <= max(x1, x2))
        & (pc_velo[:, 0] >= min(x1, x2))
    )
    return inds

def get_filename(path, with_ext=True):
    if with_ext:
        return ntpath.basename(path)
    return ntpath.basename(path).rsplit(".", 1)[0]


def get_file_path_by_idx(idx, src_dir):
    src_files = glob.glob(os.path.join(src_dir, "*"))
    for file_path in src_files:
        src_idx = get_filename(file_path, False)
        if idx == src_idx:
            return file_path
    raise ValueError("Could not find {} in {}.".format(idx, src_dir))


def pcd_to_bin(pcd, output_path):

    def bwrite(data, filename):
        with open(filename, "wb") as f:
            for x in data:
                f.write(struct.pack('f', float(x)))  # 1byte

    xyz_load = pcd
    pc_velo = xyz_load
    z = np.zeros((pc_velo.shape[0], 1), dtype=float)
    pc_velo = np.append(pc_velo, z, axis=1)
    pc_velo_flatten = list(pc_velo.reshape((1, -1))[0])
    bwrite(pc_velo_flatten, output_path)

def render_lidar_on_image(pts_velo, rgb, calib):
    # projection matrix (project from velo2cam2)
    img_height, img_width, _ = rgb.shape
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    img = np.zeros(rgb.shape)
    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(1000.0 / depth)%255, :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   1, color=tuple(color), thickness=-1)
    return img

def visualize(rgb, pc_velo, output_path):
    image = render_lidar_on_image(pc_velo, rgb, calib)
    cv2.imwrite(output_path, image)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Add disturbance to point cloud object.')
    parser.add_argument('--label', default='./labels', help='GT label dir.')
    parser.add_argument('--calib', default='./calibs', help='Calibs dir.')
    parser.add_argument('--velo', default='./velos', help='Velos dir.')
    parser.add_argument('--image', default='./images', help='Velos dir.')

    # output
    parser.add_argument('--velo_disturb', default='./disturb', help='Disturbed velo output dir.')
    parser.add_argument('--viz', default='./Viz', help='Velo visualization.')

    args = parser.parse_args()

    # make dirs
    makedirs(os.path.join(args.viz, "original"))
    makedirs(os.path.join(args.viz, "disturb"))
    makedirs(args.velo_disturb)

    # get src data dir
    src_label_dir = args.label
    src_calib_dir = args.calib
    src_velo_dir = args.velo
    src_image_dir = args.image

    # get output data dir
    output_viz_original_dir = os.path.join(args.viz, "original")
    output_viz_disturb_dir = os.path.join(args.viz, "disturb")
    output_disturb_dir = args.velo_disturb

    # default image h, w, c
    img_shape = (374, 1238, 3)

    ori_velo_paths = glob.glob(os.path.join(src_velo_dir, "*.bin"))
    # print(ori_velo_paths)
    for ori_velo_path in tqdm.tqdm(sorted(ori_velo_paths)):
        velo_idx = get_filename(ori_velo_path, False)
        calib_path = get_file_path_by_idx(velo_idx, src_calib_dir)
        label_path = get_file_path_by_idx(velo_idx, src_label_dir)
        image_path = get_file_path_by_idx(velo_idx, src_image_dir)

        output_velo_path = os.path.join(output_disturb_dir, velo_idx+".bin")
        output_viz_velo_ori_path = os.path.join(output_viz_original_dir, velo_idx+".jpg")
        output_viz_velo_disturb_path = os.path.join(output_viz_disturb_dir, velo_idx + ".jpg")

        # Load calibration
        calib = read_calib_file(calib_path)
        # Load labels
        labels = load_label(label_path)
        # Load Lidar PC
        pc_velo = load_velo_scan(ori_velo_path)[:, :3]

        proj_cam_to_velo = project_cam2_to_velo(calib)

        temp = np.asarray([[1, 1, 1]])
        delete_inds = np.asarray([0])
        for obj in labels:
            # get obj range info
            range_info = get_obj_range_info(obj)
            inds = get_obj_inds(pc_velo, range_info)
            selected = pc_velo[inds]
            selected = selected[selected[:, 2].argsort()]
            selected = get_randon_pc(selected)
            temp = np.concatenate((temp, selected), axis=0)
            delete_inds = np.concatenate((delete_inds, inds[0]), axis=0)

        new_pc_velo = np.delete(pc_velo, delete_inds[1:], axis=0)
        new_pc_velo = np.concatenate((new_pc_velo, temp[1:]), axis=0)
        pcd_to_bin(new_pc_velo, output_velo_path)

        # visualize
        rgb = cv2.imread(image_path)
        visualize(rgb, new_pc_velo, output_viz_velo_disturb_path)
        visualize(rgb, pc_velo, output_viz_velo_ori_path)


    # calib_path = "calibs/000015.txt"
    # label_path = "labels/000015.txt"
    # velo_path = "velos/000015.bin"
    # output_path = "./example.bin"
    #
    # # Pedestrian 0.00 1 0.80 189.46 158.23 256.19 356.44 1.70 0.61  0.31
    # # Load calibration
    # calib = read_calib_file(calib_path)
    #
    # # Load labels
    # labels = load_label(label_path)
    #
    # # Load Lidar PC
    # pc_velo = load_velo_scan(velo_path)[:, :3]
    #
    # proj_cam_to_velo = project_cam2_to_velo(calib)
    #
    # temp = np.asarray([[1,1,1]])
    # delete_inds = np.asarray([0])
    # for obj in labels:
    #     # get obj range info
    #     range_info = get_obj_range_info(obj)
    #     inds = get_obj_inds(pc_velo, range_info)
    #     selected = pc_velo[inds]
    #     selected = selected[selected[:, 2].argsort()]
    #     selected = get_randon_pc(selected)
    #     temp = np.concatenate((temp, selected), axis=0)
    #     delete_inds = np.concatenate((delete_inds, inds[0]), axis=0)
    #
    # pc_velo = np.delete(pc_velo, delete_inds[1:], axis=0)
    # new_pc_velo = np.concatenate((pc_velo, temp[1:]), axis=0)
    # pcd_to_bin(new_pc_velo, output_path)
    #

