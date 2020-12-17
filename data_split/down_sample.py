#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: willc
"""

"""
bin -> downsample -> bin
"""
import os
import open3d as o3d
import numpy as np
import struct
import glob
import tqdm
import ntpath

def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z, intensity])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd


def downsample(o3d_pcd, line=32):
    if line == 32:
        voxel_size = 0.07
    elif line == 16:
        voxel_size = 0.1412
    elif line == 8:
        voxel_size = 0.2547
    else:
        raise ValueError("Unsupported line number, please use one of [8, 16, 32].")
    downpcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
    return downpcd

def pcd_to_bin(pcd, output_path):

    def bwrite(data, filename):
        with open(filename, "wb") as f:
            for x in data:
                f.write(struct.pack('f', float(x)))  # 1byte

    xyz_load = np.asarray(pcd.points)
    pc_velo = xyz_load
    z = np.zeros((pc_velo.shape[0], 1), dtype=float)
    pc_velo = np.append(pc_velo, z, axis=1)
    pc_velo_flatten = list(pc_velo.reshape((1, -1))[0])
    bwrite(pc_velo_flatten, output_path)

if __name__ == "__main__":

    velo_dir_path = "/mnt/nfs/chengyong/data/kitti/8_dataset/{}/velodyne"
    dest_path = "/mnt/nfs/chengyong/data/kitti/8_dataset/{}_proc/velodyne"

    for type in ["train", "val"]:
        os.makedirs(dest_path.format(type), exist_ok=True)
        print("create dir {}".format(dest_path.format(type)))
        dir_path = velo_dir_path.format(type)
        bin_files = glob.glob(os.path.join(dir_path, "*.bin"))
        for bin_file in tqdm.tqdm(bin_files):
            file_name = ntpath.basename(bin_file)
            o3d_pcd = convert_kitti_bin_to_pcd(bin_file)
            downpcd = downsample(o3d_pcd, line=8)
            pcd_to_bin(downpcd, os.path.join(dest_path.format(type), file_name))

    print("Finished!")
