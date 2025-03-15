_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_voc20.txt',
    slide_crop=0,
    temp_thd=0.10,
    delete_same_entity=True,
    attn_rcs_weights=[2.0, 0.6],
    attn_sfr_weights=[2.1, 0.6],
)

# dataset settings
dataset_type = 'PascalVOC20Dataset'
data_root = './data/pascal_voc/VOCdevkit/VOC2012/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))
