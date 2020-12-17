import os
import ntpath
import numpy as np
import glob
from collections import defaultdict
import json
import pprint
import tqdm

def read_json(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
        return data

if __name__ == "__main__":
    anno_dir = "/mnt/nfs/chengyong/data/refinenets/20201209-1-1/Annotations"
    info_dict = defaultdict(int)
    max_x, max_y, max_z, max_w, max_h, max_l = 0, 0, 0, 0, 0, 0
    for anno_file in tqdm.tqdm(sorted(glob.glob(anno_dir+"/*.json"))):
        anno_data = read_json(anno_file)
        for img_name, anno in anno_data.items():
            # xyzwhlr
            if (anno[0] > 1): info_dict['x'] += 1
            if (anno[1] > 1): info_dict['y'] += 1
            if (anno[2] > 1): info_dict['z'] += 1
            if (anno[3] > 1): info_dict['w'] += 1
            if (anno[4] > 1): info_dict['h'] += 1
            if (anno[5] > 1): info_dict['l'] += 1
            if (anno[6] > 1): info_dict['r'] += 1

            max_x = max(max_x, anno[0])
            max_y = max(max_y, anno[1])
            max_z = max(max_z, anno[2])
            max_w = max(max_w, anno[3])
            max_h = max(max_h, anno[4])
            max_l = max(max_l, anno[5])

    
    pprint.pprint(info_dict)
    print("max x: ", max_x)
    print("max y: ", max_y)
    print("max z: ", max_z)
    print("max w: ", max_w)
    print("max h: ", max_h)
    print("max l: ", max_l)


"""
defaultdict(<class 'int'>,
            {'h': 8,
             'l': 187,
             'r': 981,
             'w': 31,
             'x': 74,
             'y': 5,
             'z': 648})

max x:  13.285632999999999
max y:  2.0665129999999996
max z:  56.572424999999996
max w:  1.351415
max h:  1.8468129999999998
max l:  3.527471
"""

