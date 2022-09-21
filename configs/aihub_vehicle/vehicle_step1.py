# step 1

# changed-file
# /root/yt/paper/mmrazor/mmrazor/models/architectures/components/backbones/searchable_shufflenet_v2.py

# changed-code

# if '300M' in model_size:
#     stage_repeats = [4, 4, 8, 4]
#     stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
# elif '1.3G' in model_size:
#     stage_repeats = [8, 8, 16, 8]
#     stage_out_channels = [-1, 48, 96, 240, 480, 960, 1024]

_base_ = ["/root/yt/paper/mmrazor/configs/nas/detnas/detnas_supernet_shufflenetv2_8xb128_in1k.py",]

# work_dir = '/root/DataSet/work_dir_nas/vehicle_step1'
work_dir = '/root/DataSet/work_dir_nas/layer100'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=20,
    train=dict(
        type='ImageNet',
        data_prefix='/root/DataSet/ImageNet/ImageNet/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=224),
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='ImageNet',
        data_prefix='/root/DataSet/ImageNet/ImageNet/ILSVRC2012_img_val',
        ann_file='/root/DataSet/ImageNet/ImageNet/caffe_ilsvrc12/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='ImageNet',
        data_prefix='/root/DataSet/ImageNet/ImageNet/ILSVRC2012_img_val',
        ann_file='/root/DataSet/ImageNet/ImageNet/caffe_ilsvrc12/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))

optimizer = dict(
    type='SGD',
    lr=0.5,
    momentum=0.9,
    weight_decay=0.00004,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
    warmup='constant',
    warmup_iters=5000,
)


runner = dict(max_iters=30000)
evaluation = dict(interval=1000, metric='accuracy')

# checkpoint saving
checkpoint_config = dict(interval=1000)

gpu_ids = range(0, 2)
