dataset_dir=/mnt/nfs/chengyong/data/kitti/16_dataset/training
label_dir=$dataset_dir/label_2
calib_dir=$dataset_dir/calib
image_dir=$dataset_dir/image_2
velo_dir=$dataset_dir/velodyne

output_dataset_dir=/mnt/nfs/chengyong/data/kitti/16_dataset_disturb/training
output_velo_dir=$output_dataset_dir/velodyne_new
output_viz_dir=$output_dataset_dir/Viz

# echo "Copy label_2"
# cp -r $label_dir $output_dataset_dir

# echo "Copy calib"
# cp -r $calib_dir $output_dataset_dir

# echo "Copy image"
# cp -r $image_dir $output_dataset_dir

# echo "Copy velo"
# cp -r $velo_dir $output_dataset_dir
# echo "Finish copying"

source /home1/chengyong/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.5

label_dir=$output_dataset_dir/label_2
calib_dir=$output_dataset_dir/calib
image_dir=$output_dataset_dir/image_2
velo_dir=$output_dataset_dir/velodyne

cd /mnt/nfs/chengyong/Workspace/thesis/disturb
python add_disturb.py \
    --label $label_dir \
    --calib $calib_dir \
    --velo $velo_dir \
    --image $image_dir \
    --velo_disturb $output_velo_dir \
    --viz $output_viz_dir
cd -

echo "Finish adding disturbance to point cloud."