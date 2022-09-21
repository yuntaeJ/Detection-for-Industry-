# step1

# python -m torch.distributed.launch \
# --nproc_per_node=2 \
# tools/mmcls/train_mmcls.py \
# configs/aihub_vehicle/vehicle_step1.py \
# --launcher pytorch

# step2

# python -m torch.distributed.launch \
# --nproc_per_node=2 \
# tools/mmdet/train_mmdet.py \
# configs/aihub_road/road_step2_large.py \
# --cfg-options load_from=/root/DataSet/work_dir_nas_road/step1/iter_300000.pth \
# --launcher pytorch

# step3

python -m torch.distributed.launch \
--nproc_per_node=2 \
tools/mmdet/search_mmdet.py \
/yt/paper/mmrazor/configs/sub_set/step3_sub_test.py \
/yt/paper/Subdataset/step2/epoch_20.pth \
--launcher pytorch

# step4

# python -m torch.distributed.launch \
# --nproc_per_node=2 \
# tools/mmcls/train_mmcls.py \
# configs/aihub_road/road_step4_large.py \
# --cfg-options algorithm.mutable_cfg=/root/DataSet/work_dir_nas_road/step3_large/final_subnet_20220423_1527.yaml \
# --launcher pytorch

# step5

# python -m torch.distributed.launch \
# --nproc_per_node=2 \
# tools/mmdet/train_mmdet.py \
# configs/aihub_road/road_step5_large.py \
# --cfg-options algorithm.mutable_cfg=/root/DataSet/work_dir_nas_road/step3_large/final_subnet_20220423_1527.yaml \
# load_from=/root/DataSet/work_dir_nas_road/step4_large/iter_150000.pth \
# --launcher pytorch


# FLOPS

# python tools/misc/get_flops.py \
# /root/DataSet/work_dir_nas_road/step5_large/road_step5_large.py





