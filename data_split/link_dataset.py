import os
import glob

if __name__ == "__main__":
    dest = "/mnt/nfs/chengyong/dev/frustum-pointnets/dataset/KITTI/object"
    src = "/mnt/nfs/chengyong/data/kitti/16_dataset/{}"

    for _type in ["training"]:
        cmd = "ln -s {} {}".format(src.format(_type), dest)
        os.system(cmd)
    
    print("Link suucessfully!")
