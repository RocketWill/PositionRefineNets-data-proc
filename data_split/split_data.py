#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: willc
"""

import ntpath
import copy
import os
import glob
import argparse
import tqdm
import time

def cp_file(idx, ext, src, dest):
    cmd = "cp {} {}".format(os.path.join(src, str(idx).zfill(6)+"."+ext), dest)
    os.system(cmd)
    time.sleep(0.01)



if __name__ == "__main__":
    src_label_dir = "/mnt/nfs/chengyong/data/kitti/training/label_2"
    src_calib_dir = "/mnt/nfs/chengyong/data/kitti/training/calib"
    src_velo_dir = "/mnt/nfs/chengyong/data/kitti/training/velodyne"
    src_image_dir = "/mnt/nfs/chengyong/data/kitti/training/image_2"

    dest_label_dir = "/mnt/nfs/chengyong/data/kitti/64_dataset/{}/label_2"
    dest_calib_dir = "/mnt/nfs/chengyong/data/kitti/64_dataset/{}/calib"
    dest_velo_dir = "/mnt/nfs/chengyong/data/kitti/64_dataset/{}/velodyne"
    dest_image_dir = "/mnt/nfs/chengyong/data/kitti/64_dataset/{}/image_2"

    train_id_path = "train_id.txt"
    val_id_path = "val_id.txt"

    with open(train_id_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            idx = (line.strip())
            cp_file(idx, "txt", src_label_dir, dest_label_dir.format("train"))
            cp_file(idx, "txt", src_calib_dir, dest_calib_dir.format("train"))
            cp_file(idx, "png", src_image_dir, dest_image_dir.format("train"))
            cp_file(idx, "bin", src_velo_dir, dest_velo_dir.format("train"))
            line = fp.readline()


    with open(val_id_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            idx = (line.strip())
            cp_file(idx, "txt", src_label_dir, dest_label_dir.format("val"))
            cp_file(idx, "txt", src_calib_dir, dest_calib_dir.format("val"))
            cp_file(idx, "png", src_image_dir, dest_image_dir.format("val"))
            cp_file(idx, "bin", src_velo_dir, dest_velo_dir.format("val"))
            line = fp.readline()