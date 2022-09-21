# step 4

dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
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
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=40,
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

paramwise_cfg = dict(bias_decay_mult=0.0, norm_decay_mult=0.0)
optimizer = dict(
    type='SGD',
    lr=0.5,
    momentum=0.9,
    weight_decay=4e-05,
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=1.0, min_lr=0.0, by_epoch=False)

evaluation = dict(interval=10000, metric='accuracy')
runner = dict(type='IterBasedRunner', max_iters=150000)
checkpoint_config = dict(interval=10000)

log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='SearchableShuffleNetV2',
        widen_factor=1.0,
        norm_cfg=dict(type='BN')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)))
mutator = dict(
    placeholder_mapping=dict(
        all_blocks=dict(
            type='OneShotOP',
            choices=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock',
                    kernel_size=3,
                    norm_cfg=dict(type='BN')),
                shuffle_5x5=dict(
                    type='ShuffleBlock',
                    kernel_size=5,
                    norm_cfg=dict(type='BN')),
                shuffle_7x7=dict(
                    type='ShuffleBlock',
                    kernel_size=7,
                    norm_cfg=dict(type='BN')),
                shuffle_xception=dict(
                    type='ShuffleXception', norm_cfg=dict(type='BN'))))))
algorithm = dict(
    type='SPOS',
    architecture=dict(
        type='MMClsArchitecture',
        model=dict(
            type='mmcls.ImageClassifier',
            backbone=dict(
                type='SearchableShuffleNetV2',
                widen_factor=1.0,
                norm_cfg=dict(type='BN')),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=1024,
                loss=dict(
                    type='LabelSmoothLoss',
                    num_classes=1000,
                    label_smooth_val=0.1,
                    mode='original',
                    loss_weight=1.0),
                topk=(1, 5)))),
    mutator=dict(
        type='OneShotMutator',
        placeholder_mapping=dict(
            all_blocks=dict(
                type='OneShotOP',
                choices=dict(
                    shuffle_3x3=dict(
                        type='ShuffleBlock',
                        kernel_size=3,
                        norm_cfg=dict(type='BN')),
                    shuffle_5x5=dict(
                        type='ShuffleBlock',
                        kernel_size=5,
                        norm_cfg=dict(type='BN')),
                    shuffle_7x7=dict(
                        type='ShuffleBlock',
                        kernel_size=7,
                        norm_cfg=dict(type='BN')),
                    shuffle_xception=dict(
                        type='ShuffleXception', norm_cfg=dict(type='BN')))))),
    distiller=None,
    retraining=True,
    mutable_cfg=''
)
find_unused_parameters = False
work_dir = '/root/DataSet/work_dir_nas/vehicle_step4_large'

