#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: willc
"""

"""
输入：
    1. gt file
    2. predicts file
输出：
    target file (base on iou)
"""


import numpy as np
from utils import obj_to_8_points, read_label, Object3d, read_calib_file, get_xyzwhlrs_offset, \
    crop_image, get_box_range, obj_to_8_points, draw_projected_box3d, write_json, \
    load_velo_scan, render_lidar_on_image
from collections import defaultdict, OrderedDict
import cv2
import ntpath
import pprint
import copy
import os
import glob
import argparse
import tqdm


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

def visualize_match_result(image_path, match_dict):
    """
    :param image_path:
    :param match_dict: {Object3d: Object3d....}
    :return: cv image
    """
    image = cv2.imread(image_path)
    for pred, gt in match_dict.items():
        if pred != None:
            pred_box = list(pred.box2d)
            cv2.rectangle(image, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (0, 255, 0), 2)
        if gt != None:
            gt_box = list(gt.box2d)
            cv2.rectangle(image, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])),
                          (0, 0, 255), 2)
    return image

def calculate_box_plot_thickness(box):
    """
    :param box: [[x0, y0], [x1, y1]]
    :return: thickness
    """
    box = np.asarray(box)
    width, height = box[1] - box[0]
    area = width * height
    thickness = 1
    if area > 5000:
        thickness = 2
    if area > 20000:
        thickness = 3
    return thickness


def get_filename(path, with_ext=True):
    if with_ext:
        return ntpath.basename(path)
    return ntpath.basename(path).rsplit(".", 1)[0]


def makedirs(path):
    os.makedirs(path, exist_ok=True)
    print("Make new dir: {}".format(path))

def get_file_path_by_idx(idx, src_dir):
    src_files = glob.glob(os.path.join(src_dir, "*"))
    for file_path in src_files:
        src_idx = get_filename(file_path, False)
        if idx == src_idx:
            return file_path
    raise ValueError("Could not find {} in {}.".format(idx, src_dir))

def get_match_objs(pred_objects, gt_objects):
    match = OrderedDict()
    for idx, pred_obj in enumerate(pred_objects):
        if pred_obj.score < obj_score_thresh:
            # print("Score too low, skip.")
            continue

        max_score = 0
        interested_obj = None
        pred_box = list(pred_obj.box2d)
        for gt_obj in gt_objects:
            if gt_obj.type == "DontCare":
                # print("DontCare, skip.")
                continue
            gt_box = list(gt_obj.box2d)
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_thresh:
                max_score = iou
                interested_obj = gt_obj

        match[pred_obj] = interested_obj
    return match

def gen_offset(prefix_name, match, offset_output_dir):
    idx = 1
    offset_dict = OrderedDict()
    for pred_obj, gt_obj in match.items():
        if gt_obj == None:
            # print("Pred obj does not match any gt, skip.")
            continue
        pred_obj_info = pred_obj.get_obj_xyzwhlr()
        gt_obj_info = gt_obj.get_obj_xyzwhlr()
        offset = get_xyzwhlrs_offset(pred_obj_info, gt_obj_info)
        offset_dict[prefix_name + "_" + str(idx).zfill(3)] = list(offset)
        idx += 1
    # write to file
    # os.makedirs(offset_output_dir, exist_ok=True)
    if offset_dict:
        write_json(os.path.join(offset_output_dir, "{}.json".format(prefix_name)), offset_dict)
        return True
    else:
        return False

def crop_rgb_imgs(prefix_name, match_8_pts, image, rgb_imgs_output_path):
    idx = 1
    for i in range(0, len(match_8_pts), 2):
        img_clone = copy.deepcopy(image)
        pred_obj_pts = match_8_pts[i]
        img_name = prefix_name + "_" + str(idx).zfill(3) + ".jpg"
        try:
            box = get_box_range(pred_obj_pts)
        except:
            idx += 1
            # print("Invalid object!")
            continue

        thickness = calculate_box_plot_thickness(box)
        img = draw_projected_box3d(img_clone, pred_obj_pts, color=(0, 255, 0), thickness=thickness)
        crop = crop_image(img, box)
        cv2.imwrite(os.path.join(rgb_imgs_output_path, img_name), crop)
        idx += 1

def crop_depth_imgs(prefix_name, image, depth_imgs_output_path, velo_path):
    rgb = copy.deepcopy(image)
    img_height, img_width, img_channel = rgb.shape
    pc_velo = load_velo_scan(velo_path)[:, :3]
    depth_image = render_lidar_on_image(pc_velo, rgb, calibs, img_width, img_height)

    idx = 1
    for i in range(0, len(match_8_pts), 2):
        img_clone = copy.deepcopy(depth_image)
        pred_obj_pts = match_8_pts[i]
        img_name = prefix_name + "_" + str(idx).zfill(3) + ".jpg"
        try:
            box = get_box_range(pred_obj_pts)
        except:
            idx += 1
            print("Invalid object!")
            continue
        thickness = calculate_box_plot_thickness(box)
        img = draw_projected_box3d(img_clone, pred_obj_pts, color=(255, 255, 255), thickness=thickness)
        crop = crop_image(img, box)
        cv2.imwrite(os.path.join(depth_imgs_output_path, img_name), crop)
        idx += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare Refinets Dataset')
    parser.add_argument('--image', default='./images', help='RGB images dir.')
    parser.add_argument('--label', default='./labels', help='GT label dir.')
    parser.add_argument('--pred', default='./predicts', help='Predict label dir.')
    parser.add_argument('--calib', default='./calibs', help='Calibs dir.')
    parser.add_argument('--velo', default='./velos', help='Velos dir.')

    # output
    parser.add_argument('--anno', default='./Annotations', help='Annotations output dir.')
    parser.add_argument('--imgdata', default='./Img', help='Images dataset output dir.')
    parser.add_argument('--viz', default='./Viz', help='IoU visualize dir.')

    args = parser.parse_args()

    # make dirs
    makedirs(args.anno)
    makedirs(args.viz)
    makedirs(os.path.join(args.imgdata, "rgb"))
    makedirs(os.path.join(args.imgdata, "depth"))

    # get src data dir
    src_img_dir = args.image
    src_label_dir = args.label
    src_pred_dir = args.pred
    src_calib_dir = args.calib
    src_velo_dir = args.velo

    # get output data dir
    output_anno_dir = args.anno
    output_viz_dir = args.viz
    output_rgb_dir = os.path.join(args.imgdata, "rgb")
    output_depth_dir = os.path.join(args.imgdata, "depth")

    # threshholds
    obj_score_thresh = 0.2
    iou_thresh = 0.5


    # gt_file = "data/000015_gt.txt"
    # pred_file = "data/000015_pred.txt"
    # img_path = "data/000015.png"
    # calib_path = "data/000015_calib.txt"
    # velo_path = "data/000015_velo.bin"
    #
    # output_viz_path = "data/match.jpg"
    # crop_imgs_output_dir = "Images"
    # offset_output_dir = "Annotations"
    #
    # gt_objects = read_label(gt_file)
    # pred_objects = read_label(pred_file)

    pred_paths = glob.glob(os.path.join(src_pred_dir, "*.txt"))
    for idx, pred_path in tqdm.tqdm(enumerate(pred_paths)):
        pred_idx = get_filename(pred_path, False)
        gt_path = get_file_path_by_idx(pred_idx, src_label_dir)

        # Get matched objects
        gt_objects = read_label(gt_path)
        pred_objects = read_label(pred_path)
        match = get_match_objs(pred_objects, gt_objects)

        # Generate target offset
        is_valid = gen_offset(pred_idx, match, output_anno_dir)
        if not is_valid:
            print("without detected objs, skip.")
            continue

        calib_path = get_file_path_by_idx(pred_idx, src_calib_dir)
        calibs = read_calib_file(calib_path)
        P = calibs["P2"]
        P = np.reshape(P, [3, 4])
        match_8_pts = obj_to_8_points(match, P)

        # Generate crop images
        img_path = get_file_path_by_idx(pred_idx, src_img_dir)
        image = cv2.imread(img_path)

        # viz iou match
        viz_image = visualize_match_result(img_path, match)
        cv2.imwrite(os.path.join(output_viz_dir, "./{}.jpg".format(pred_idx)), viz_image)
        # print("output image")

        # RGB
        output_rgb_idx_path = os.path.join(output_rgb_dir, pred_idx)
        os.makedirs(output_rgb_idx_path, exist_ok=True)
        crop_rgb_imgs(pred_idx, match_8_pts, image, output_rgb_idx_path)

        # Depth
        velo_path = get_file_path_by_idx(pred_idx, src_velo_dir)
        output_depth_idx_path = os.path.join(output_depth_dir, pred_idx)
        os.makedirs(output_depth_idx_path, exist_ok=True)
        crop_depth_imgs(pred_idx, image, output_depth_idx_path, velo_path)

