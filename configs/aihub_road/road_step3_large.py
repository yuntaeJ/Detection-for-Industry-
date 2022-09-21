
# step 3

# _base_=[
#     '/root/yt/paper/mmrazor/configs/nas/detnas/detnas_supernet_frcnn_shufflenetv2_fpn_1x_coco.py'
# ]


classes_list = ('1', '2', '3', '4', '5', '10')
class_num = 6

work_dir = '/root/DataSet/work_dir_nas_road/step3_large'

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=40,
    train=dict(
        type='CocoDataset',
        classes=classes_list,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        ann_file=
        '/root/DataSet/aihub_road/annotations/road_train.json',
        img_prefix='/root/DataSet/aihub_road/images'),
    val=dict(
        type='CocoDataset',
        classes=classes_list,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        ann_file=
        '/root/DataSet/aihub_road/annotations/road_val.json',
        img_prefix='/root/DataSet/aihub_road/images'),
    test=dict(
        type='CocoDataset',
        classes=classes_list,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        ann_file=
        '/root/DataSet/aihub_road/annotations/road_val.json',
        img_prefix='/root/DataSet/aihub_road/images'))
evaluation = dict(interval=1, metric='bbox', classwise = True)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ''
resume_from = None
workflow = [('train', 1)]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='mmdet.FasterRCNN',
    backbone=dict(
        type='mmcls.SearchableShuffleNetV2',
        norm_cfg=dict(type='BN', requires_grad=True),
        out_indices=(0, 1, 2, 3),
        widen_factor=1.0,
        with_last_layer=False),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN', requires_grad=True),
        in_channels=[96, 240, 480, 960],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            norm_cfg=dict(type='BN', requires_grad=True),
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=class_num,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
mutator = dict(
    type='OneShotMutator',
    placeholder_mapping=dict(
        all_blocks=dict(
            type='OneShotOP',
            choices=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock',
                    norm_cfg=dict(type='BN', requires_grad=True),
                    kernel_size=3),
                shuffle_5x5=dict(
                    type='ShuffleBlock',
                    norm_cfg=dict(type='BN', requires_grad=True),
                    kernel_size=5),
                shuffle_7x7=dict(
                    type='ShuffleBlock',
                    norm_cfg=dict(type='BN', requires_grad=True),
                    kernel_size=7),
                shuffle_xception=dict(
                    type='ShuffleXception',
                    norm_cfg=dict(type='BN', requires_grad=True))))))
algorithm = dict(
    type='DetNAS',
    bn_training_mode=True, ### for step 3
    architecture=dict(
        type='MMDetArchitecture',
        model=dict(
            type='mmdet.FasterRCNN',
            backbone=dict(
                type='mmcls.SearchableShuffleNetV2',
                norm_cfg=dict(type='BN', requires_grad=True),
                out_indices=(0, 1, 2, 3),
                widen_factor=1.0,
                with_last_layer=False),
            neck=dict(
                type='FPN',
                norm_cfg=dict(type='BN', requires_grad=True),
                in_channels=[96, 240, 480, 960],
                out_channels=256,
                num_outs=5),
            rpn_head=dict(
                type='RPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            roi_head=dict(
                type='StandardRoIHead',
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(
                        type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]),
                bbox_head=dict(
                    type='Shared4Conv1FCBBoxHead',
                    norm_cfg=dict(type='BN', requires_grad=True),
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=class_num,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=False,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
            train_cfg=dict(
                rpn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.3,
                        match_low_quality=True,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.5,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=False),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False),
                rpn_proposal=dict(
                    nms_pre=2000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0),
                rcnn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False)),
            test_cfg=dict(
                rpn=dict(
                    nms_pre=1000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0),
                rcnn=dict(
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100)))),
    mutator=dict(
        type='OneShotMutator',
        placeholder_mapping=dict(
            all_blocks=dict(
                type='OneShotOP',
                choices=dict(
                    shuffle_3x3=dict(
                        type='ShuffleBlock',
                        norm_cfg=dict(type='BN', requires_grad=True),
                        kernel_size=3),
                    shuffle_5x5=dict(
                        type='ShuffleBlock',
                        norm_cfg=dict(type='BN', requires_grad=True),
                        kernel_size=5),
                    shuffle_7x7=dict(
                        type='ShuffleBlock',
                        norm_cfg=dict(type='BN', requires_grad=True),
                        kernel_size=7),
                    shuffle_xception=dict(
                        type='ShuffleXception',
                        norm_cfg=dict(type='BN', requires_grad=True)))))),
    pruner=None,
    distiller=None,
    retraining=False)
find_unused_parameters = True
gpu_ids = range(0, 2)

# ################### evolution search #######################

#  algorithm = dict(bn_training_mode=True)

searcher = dict(
    type='EvolutionSearcher',
    metrics='bbox',
    score_key='bbox_mAP',
    constraints=dict(flops=300 * 1e8),
    candidate_pool_size=50,
    candidate_top_k=10,
    max_epoch=20,
    num_mutation=20,
    num_crossover=20,
)
