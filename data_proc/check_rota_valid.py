import os
import ntpath
import numpy as np
import glob
from collections import defaultdict
import json
import pprint
import tqdm
import pprint
from utils import obj_to_8_points, read_label, Object3d, read_calib_file, get_xyzwhlrs_offset, \
    crop_image, get_box_range, obj_to_8_points, draw_projected_box3d, write_json, \
    load_velo_scan, render_lidar_on_image

if __name__ == "__main__":
    pred_dir = "/mnt/nfs/chengyong/dev/frustum-pointnets/train/detection_results_16_v3/data"

    valid_dict = defaultdict(int)
    max_ry = -1000
    min_ry = 1000
    for pred_file in tqdm.tqdm(sorted(glob.glob(pred_dir+"/*.txt"))):
        pred_objects = read_label(pred_file)

        for obj in pred_objects:
            ry = obj.ry
            max_ry = max(max_ry, ry)
            min_ry = min(min_ry, ry)
            if ry > 3.14 or ry < -3.14:
                valid_dict[ntpath.basename(pred_file)] += 1

    pprint.pprint(valid_dict)
    print("max ry: ", max_ry)
    print("min ry: ", min_ry)


    info_dict = defaultdict(int)
    pred_dir = "/mnt/nfs/chengyong/dev/thesis/TDPosRefineNet_v2_2/line-16-v3_val_iter15000_2"
    for pred_file in tqdm.tqdm(sorted(glob.glob(pred_dir+"/*.txt"))):
        pred_objects = read_label(pred_file)

        for obj in pred_objects:
            # if obj.w <0 or obj.h < 0 or obj.l < 0:
            #     info_dict[pred_file] += 1
            if obj.w < 0. or obj.h < 0. or obj.l < 0.:
                info_dict[pred_file] += 1
    pprint.pprint(info_dict)
    print(len(glob.glob(pred_dir+"/*.txt")))
    print(len(info_dict.keys()))