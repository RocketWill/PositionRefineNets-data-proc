import numpy as np
import os
from PIL import Image, ImageDraw

DATASET_PATH = "<DATASET PATH HERE>"


import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import *


def render_image_with_boxes(img, objects, calib):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    img1 = np.copy(img)
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord)

    plt.imshow(img1)
    plt.yticks([])
    plt.xticks([])
    plt.show()


def render_lidar_with_boxes(pc_velo, objects, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]
    imgfov_pc_velo = pc_velo[inds, :]

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))

    draw_lidar(imgfov_pc_velo, fig=fig)

    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        # Project boxes from camera to lidar coordinate
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)

        # Draw boxes
        draw_gt_boxes3d(boxes3d_pts, fig=fig)
    mlab.show()


def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    # print(len(pc_velo))
    # print(len(pts_velo))
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)
    # print(pts_2d)
    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]



    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # downsample
    # a = imgfov_pc_pixel[0][::2]
    # b = imgfov_pc_pixel[1][::2]
    #
    # print(imgfov_pc_pixel)
    # imgfov_pc_pixel = np.array([a, b])

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]

    # print(imgfov_pc_velo)

    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))

    # print(imgfov_pc_velo[100:])
    # ccc = split_list(imgfov_pc_velo, len(imgfov_pc_velo) // 64)
    # pc_velo = np.asarray(subsample(ccc))




    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    img = np.zeros(img.shape)
    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(1000.0 / depth)%255, :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   1, color=tuple(color), thickness=-1)
    # plt.imshow(img)
    # plt.yticks([])
    # plt.xticks([])
    # plt.show()
    return img

def split_list(list, seg_length):
    inlist = list[:, :]
    outlist = []

    while len(inlist):
        # print(inlist)
        outlist.append(inlist[0:seg_length, :])
        # print(inlist[0:seg_length, :])
        # inlist = inlist[0:seg_length, :]
        inlist = inlist[seg_length:, :]

    return outlist

def subsample(velo_group):
    res = []
    for i, v in enumerate(velo_group):
        if i%2 == 0:
            for x in v:
                res.append(x)
    return res

if __name__ == '__main__':
    # Load image, calibration file, label bbox
    calib_path = "calibs/000015.txt"
    label_path = "labels/000015.txt"
    velo_path = "velos/000015.bin"
    image_path = "images/000015.png"


    rgb = cv2.cvtColor(cv2.imread(os.path.join(image_path)), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape
    print(rgb.shape)

    # Load calibration
    calib = read_calib_file(calib_path)

    # Load labels
    labels = load_label(label_path)

    # Load Lidar PC
    # pc_velo = load_velo_scan('/Volumes/Will 1/thesis/cus_ktti_vis/000073.bin')[:, :3]
    pc_velo = load_velo_scan(velo_path)[:, :3]
    # pc_velo = load_velo_scan('example.bin')
    print(pc_velo)
    pc_velo = pc_velo[pc_velo[:, 2].argsort()]
    print(len(pc_velo))

    # import open3d as o3d
    # pcd_load = o3d.io.read_point_cloud("hello.pcd")
    # xyz_load = np.asarray(pcd_load.points)
    # pc_velo = xyz_load



    ccc = split_list(pc_velo, len(pc_velo)//64)
    pc_velo = np.asarray(subsample(ccc))
    # pc_velo = pc_velo[::4, :]

    # pc_velo = load_velo_scan('/Volumes/Will 1/thesis/cus_ktti_vis/data_split/example.bin')[:, :3]

    pc_velo = load_velo_scan('/Users/willc/Desktop/cus_ktti_vis/disturb/example.bin')[:, :3]

    pc_velo = load_velo_scan(velo_path)[:, :3]
    print(pc_velo)

    # render_image_with_boxes(rgb, labels, calib)
    # render_lidar_with_boxes(pc_velo, labels, calib, img_width=img_width, img_height=img_height)
    image = render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)
    cv2.imwrite('./z-16-test_o.jpg', image)