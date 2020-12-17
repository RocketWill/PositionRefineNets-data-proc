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
    if thickness > 20000:
        thickness = 3
    return thickness


if __name__ == "__main__":

    gt_file = "data/000015_gt.txt"
    pred_file = "data/000015_pred.txt"
    img_path = "data/000015.png"
    calib_path = "data/000015_calib.txt"
    velo_path = "data/000015_velo.bin"

    output_viz_path = "data/match.jpg"
    crop_imgs_output_dir = "Images"
    offset_output_dir = "Annotations"

    gt_objects = read_label(gt_file)
    pred_objects = read_label(pred_file)

    obj_score_thresh = 0.2
    iou_thresh = 0.5

    match = OrderedDict()


    for idx, pred_obj in enumerate(pred_objects):
        if pred_obj.score < obj_score_thresh:
            print("Score too low, skip.")
            continue

        max_score = 0
        interested_obj = None
        pred_box = list(pred_obj.box2d)
        for gt_obj in gt_objects:
            if gt_obj.type == "DontCare":
                print("DontCare, skip.")
                continue
            gt_box = list(gt_obj.box2d)
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_thresh:
                max_score = iou
                interested_obj = gt_obj

        match[pred_obj] = interested_obj

    viz_image = visualize_match_result(img_path, match)
    cv2.imwrite(output_viz_path, viz_image)


    """
    Generate target offset
    """
    idx = 1
    prefix_name = ntpath.basename(img_path).rsplit('.', 1)[0]
    offset_dict = OrderedDict()
    for pred_obj, gt_obj in match.items():
        if gt_obj == None:
            print("Pred obj does not match any gt, skip.")
            continue
        pred_obj_info = pred_obj.get_obj_xyzwhlr()
        gt_obj_info = gt_obj.get_obj_xyzwhlr()
        offset = get_xyzwhlrs_offset(pred_obj_info, gt_obj_info)
        offset_dict[prefix_name + "_" + str(idx).zfill(3)] = list(offset)
        idx += 1
    # write to file
    os.makedirs(offset_output_dir, exist_ok=True)
    write_json(os.path.join(offset_output_dir, "annotation.json"), offset_dict)


    """
    Generate crop image
    """
    # RGB image
    calibs = read_calib_file(calib_path)
    P = calibs["P2"]
    P = np.reshape(P, [3, 4])
    match_8_pts = obj_to_8_points(match, P)
    image = cv2.imread(img_path)
    os.makedirs(crop_imgs_output_dir, exist_ok=True)
    idx = 1
    rgb_imgs_output_path = os.path.join(crop_imgs_output_dir, "rgb")
    os.makedirs(rgb_imgs_output_path, exist_ok=True)
    for i in range(0, len(match_8_pts), 2):
        img_clone = copy.deepcopy(image)
        pred_obj_pts = match_8_pts[i]
        img_name = prefix_name + "_" + str(idx).zfill(3) + ".jpg"
        box = get_box_range(pred_obj_pts)

        thickness = calculate_box_plot_thickness(box)
        img = draw_projected_box3d(img_clone, pred_obj_pts, color=(0,255,0), thickness=thickness)
        crop = crop_image(img, box)
        cv2.imwrite(os.path.join(rgb_imgs_output_path, img_name), crop)
        idx += 1

    # Depth image
    rgb = cv2.imread(img_path)
    img_height, img_width, img_channel = rgb.shape
    pc_velo = load_velo_scan(velo_path)[:, :3]
    depth_image = render_lidar_on_image(pc_velo, rgb, calibs, img_width, img_height)

    idx = 1
    depth_imgs_output_path = os.path.join(crop_imgs_output_dir, "depth")
    os.makedirs(depth_imgs_output_path, exist_ok=True)
    for i in range(0, len(match_8_pts), 2):
        img_clone = copy.deepcopy(depth_image)
        pred_obj_pts = match_8_pts[i]
        img_name = prefix_name + "_" + str(idx).zfill(3) + ".jpg"
        box = get_box_range(pred_obj_pts)

        thickness = calculate_box_plot_thickness(box)
        img = draw_projected_box3d(img_clone, pred_obj_pts, color=(255,255,255), thickness=thickness)
        crop = crop_image(img, box)
        cv2.imwrite(os.path.join(depth_imgs_output_path, img_name), crop)
        idx += 1

