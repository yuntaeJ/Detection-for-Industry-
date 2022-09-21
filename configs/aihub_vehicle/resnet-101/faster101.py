_base_ = '/root/yt/paper/mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py'

classes_list = ('person','mask')
class_num = 2

data = dict(
    samples_per_gpu=10,
    workers_per_gpu=40,
    train=dict(
        classes = classes_list,
        ann_file=
        '/root/DataSet/crowdhuman/annotations/worker_train.json', 
        img_prefix='/root/DataSet/crowdhuman/Images'),
    val=dict(
        classes = classes_list,
        ann_file=
        '/root/DataSet/crowdhuman/annotations/worker_val.json',
        img_prefix='/root/DataSet/crowdhuman/Images'),
    test=dict(
        classes = classes_list,
        ann_file=
        '/root/DataSet/crowdhuman/annotations/worker_val.json',
        img_prefix='/root/DataSet/crowdhuman/Images'))

work_dir = '/root/DataSet/experiments/worker_resnet101'

evaluation = dict(interval=1, metric='bbox', classwise=True)

model=dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=class_num)))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict (grad_clip = dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict( 
    policy='cyclic', 
    target_ratio=(2e-2, 1e-4), 
    cyclic_times=8, 
    step_ratio_up=0.4) 
momentum_config = dict( 
    policy='cyclic', 
    target_ratio=(0.85 / 0.95, 1), 
    cyclic_times=8, 
    step_ratio_up=0.4)


checkpoint_config = dict(interval=4)

runner = dict(type='EpochBasedRunner', max_epochs=100)
resume_from = '/root/DataSet/experiments/worker_resnet101/epoch_32.pth'