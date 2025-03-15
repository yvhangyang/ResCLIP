_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_city_scapes.txt',
    temp_thd=0.10,
    delete_same_entity=True,
    attn_rcs_weights=[2.0, 0.2],
    attn_sfr_weights=[2.1, 0.6],
)

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = './data/OpenDataLab___CityScapes/raw'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 560), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth does not need to do resize data transform
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
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
