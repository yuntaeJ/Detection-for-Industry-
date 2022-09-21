python -m torch.distributed.launch \
--nproc_per_node=2 \
tools/mmdet/search_mmdet.py \
configs/sub_set/step3_sub_test.py \ 
work_dirs/step2/epoch_20.pth \
--launcher pytorch