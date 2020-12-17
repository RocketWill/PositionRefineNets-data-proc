#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: willc
"""
import numpy as np
import cv2
import random

def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t

def pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.zeros((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t

def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    # return image.transpose(2, 0, 1)
    return image


if __name__ == "__main__":
    img_path = "/Volumes/Will 1/thesis/cus_ktti_vis/data_proc/Img/rgb/000015/000015_005.jpg"
    img = cv2.imread(img_path)

    pad_img = pad_to_square(img, (104, 117, 123), True)
    pad_img = _resize_subtract_mean(pad_img, 1000, (104, 117, 123))
    cv2.imwrite("./pad.jpg", pad_img)
