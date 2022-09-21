# The new config inherits a base config to highlight the necessary modification
_base_ = '/root/yt/paper/mmdetection/configs/tood/tood_r50_fpn_mstrain_2x_coco.py'


# Num class
model = dict(
    bbox_head=dict(num_classes=2))

# Data
# Modify dataset related settings
dataset_type = 'COCODataset'
classes_list = ('person','mask')
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=20,
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
# Training schedule
runner = dict(type='EpochBasedRunner', max_epochs=30)

# Evaluation times
evaluation = dict(interval=1, metric='bbox',
                 classwise = True)

# Check point save
checkpoint_config = dict(interval=1)

# Workflow
workflow = [('train', 1), ('val', 1)]
work_dir = './work_dirs/tood'

