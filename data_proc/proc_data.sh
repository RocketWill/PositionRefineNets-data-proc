src_image_dir=/mnt/nfs/chengyong/dev/frustum-pointnets/dataset/KITTI/object/training/image_2
src_label_dir=/mnt/nfs/chengyong/dev/frustum-pointnets/dataset/KITTI/object/training/label_2
src_calib_dir=/mnt/nfs/chengyong/dev/frustum-pointnets/dataset/KITTI/object/training/calib
src_velo_dir=/mnt/nfs/chengyong/dev/frustum-pointnets/dataset/KITTI/object/training/velodyne

src_pred_label_dir=/mnt/nfs/chengyong/dev/frustum-pointnets/train/detection_results_16_v3/data

dest_anno_dir=/mnt/nfs/chengyong/data/refinenets/20201214-1-1/Annotations
dest_image_dir=/mnt/nfs/chengyong/data/refinenets/20201214-1-1/Images
dest_viz_dir=/mnt/nfs/chengyong/data/refinenets/20201214-1-1/viz

python prepare_data.py --image $src_image_dir \
--label $src_label_dir --pred $src_pred_label_dir \
--calib $src_calib_dir --velo $src_velo_dir \
--anno $dest_anno_dir --imgdata $dest_image_dir --viz $dest_viz_dir

