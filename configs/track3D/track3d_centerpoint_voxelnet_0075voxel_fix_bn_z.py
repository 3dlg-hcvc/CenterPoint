import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=1, class_names=["object"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=6,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=6, ds_factor=8
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='track3d',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=False
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=1,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

# test_cfg = dict(
#     post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#     max_per_img=500,
#     nms=dict(
#         use_rotate_nms=True,
#         use_multi_class_nms=False,
#         nms_pre_max_size=1000,
#         nms_post_max_size=83,
#         nms_iou_threshold=0.2,
#     ),
#     score_threshold=0.1,
#     pc_range=[-54, -54],
#     out_size_factor=get_downsample_factor(model),
#     voxel_size=[0.075, 0.075]
# )

# dataset settings
dataset_type = "Track3DDataset"
data_root = "data/track3D"

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

# Keep the voxel number 1440 * 1440 * 40
voxel_generator = dict(
    range=[-9.0, -3.0, -5.0, 9.0, 15.0, 11.0],
    voxel_size=[0.0125, 0.0125, 0.4],
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
)

# TODO: Update the train pipeline to get our data
train_pipeline = [
    dict(type="Track3DVoxelization", cfg=voxel_generator),
    dict(type="Track3DAssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Track3DReformat"),
]
# test_pipeline = [
#     dict(type="LoadPointCloudFromFile", dataset=dataset_type),
#     dict(type="LoadPointCloudAnnotations", with_bbox=True),
#     dict(type="Preprocess", cfg=val_preprocessor),
#     dict(type="Voxelization", cfg=voxel_generator),
#     dict(type="AssignLabel", cfg=train_cfg["assigner"]),
#     dict(type="Reformat"),
# ]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        root_path=f"{data_root}/train",
        class_names=class_names,
        pipeline=train_pipeline,
        shuffle_points=True,
        mode="train"
    ),
    # val=dict(
    #     type=dataset_type,
    #     root_path=data_root,
    #     test_mode=True,
    #     class_names=class_names,
    #     pipeline=test_pipeline,
    #     shuffle_points=False,
    #     mode="val"
    # ),
    # test=dict(
    #     type=dataset_type,
    #     root_path=data_root,
    #     test_mode=True,
    #     class_names=class_names,
    #     pipeline=test_pipeline,
    # ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]
