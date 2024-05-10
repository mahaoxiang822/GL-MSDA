norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='FasterRCNNOBBRGBDDDDepth',
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    dump_folder='eval/sim_to_real/rgb_ddd_sim',
    fusion_type='concat_final',
    rgb_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe'),
    depth_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe'),
    rpn_head=dict(
        type='RPNHeadGraspNet',
        in_channels=1024,
        feat_channels=1024,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16],
            ratios=[0.5, 1.0, 2.0],
            strides=[16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHeadOBBRGBD',
        shared_head=dict(
            type='ResLayer',
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='BBoxHeadGraspNetDepth',
            in_channels=2048,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYZWHSinCosGraspCoder',
                target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0],
                target_stds=[0.1, 0.1, 0.2, 0.2, 1.0, 1.0, 10.0]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_score=dict(type='SmoothL1Loss', beta=1.0, loss_weight=4.0),
            loss_grasp=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxScoreAssignerGraspNet',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxScoreAssignerGraspNet',
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
            nms_pre=6000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(score_threshold=0.05, camera='realsense')))
train_dataset_type = 'PybulletRandomDataset'
test_dataset_type = 'GraspNetDataset'
train_data_root = 'data/pybullet_random/'
test_data_root = 'data/planer_graspnet'
rgb_norm_cfg = dict(
    mean=[105.4, 118.08, 115.29], std=[71.78, 67.65, 70.59], to_rgb=True)
depth_norm_cfg = dict(mean=477.11, std=101.36)
train_pipeline = [
    dict(
        type='LoadRGBDepthGraspNet',
        with_rgb=True,
        with_depth=True,
        with_origin_depth=True),
    dict(type='LoadAnnotationsGraspNet'),
    dict(
        type='ResizeGraspNet',
        with_rgb=True,
        with_depth=True,
        with_origin_depth=True,
        img_scale=(1280, 720),
        keep_ratio=True),
    dict(
        type='RandomFlipGraspNet',
        with_rgb=True,
        with_depth=True,
        with_origin_depth=True,
        flip_ratio=0.5),
    dict(
        type='RandomRotateGraspNet',
        rotate_ratio=0.5,
        with_rgb=True,
        with_depth=True,
        with_origin_depth=True,
        angle=(-180, 180)),
    dict(
        type='NormalizeRGB',
        mean=[105.4, 118.08, 115.29],
        std=[71.78, 67.65, 70.59],
        to_rgb=True),
    dict(type='NormalizeDepth', mean=477.11, std=101.36),
    dict(
        type='PadGraspNet',
        with_rgb=True,
        with_depth=True,
        with_origin_depth=True,
        size_divisor=32),
    dict(type='DefaultFormatBundleGraspNet'),
    dict(
        type='CollectGraspNet',
        keys=[
            'rgb', 'depth', 'origin_depth', 'gt_rect_grasps', 'gt_scores',
            'gt_object_ids', 'gt_depths'
        ])
]
test_pipeline = [
    dict(
        type='LoadRGBDepthGraspNet',
        with_rgb=True,
        with_depth=True,
        with_origin_depth=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(
                type='ResizeGraspNet',
                with_rgb=True,
                with_depth=True,
                with_origin_depth=True,
                keep_ratio=True),
            dict(
                type='NormalizeRGB',
                mean=[105.4, 118.08, 115.29],
                std=[71.78, 67.65, 70.59],
                to_rgb=True),
            dict(type='NormalizeDepth', mean=477.11, std=101.36),
            dict(
                type='PadGraspNet',
                with_rgb=True,
                with_depth=True,
                with_origin_depth=True,
                size_divisor=32),
            dict(type='ImageToTensor', keys=['rgb', 'depth', 'origin_depth']),
            dict(
                type='CollectGraspNet', keys=['rgb', 'depth', 'origin_depth'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='PybulletRandomDataset',
        root='data/pybullet_random/',
        camera='realsense',
        rect_label_folder='rect_labels_filt_nms_0.02_10',
        split='train',
        view='1016',
        pipeline=[
            dict(
                type='LoadRGBDepthGraspNet',
                with_rgb=True,
                with_depth=True,
                with_origin_depth=True),
            dict(type='LoadAnnotationsGraspNet'),
            dict(
                type='ResizeGraspNet',
                with_rgb=True,
                with_depth=True,
                with_origin_depth=True,
                img_scale=(1280, 720),
                keep_ratio=True),
            dict(
                type='RandomFlipGraspNet',
                with_rgb=True,
                with_depth=True,
                with_origin_depth=True,
                flip_ratio=0.5),
            dict(
                type='RandomRotateGraspNet',
                rotate_ratio=0.5,
                with_rgb=True,
                with_depth=True,
                with_origin_depth=True,
                angle=(-180, 180)),
            dict(
                type='NormalizeRGB',
                mean=[105.4, 118.08, 115.29],
                std=[71.78, 67.65, 70.59],
                to_rgb=True),
            dict(type='NormalizeDepth', mean=477.11, std=101.36),
            dict(
                type='PadGraspNet',
                with_rgb=True,
                with_depth=True,
                with_origin_depth=True,
                size_divisor=32),
            dict(type='DefaultFormatBundleGraspNet'),
            dict(
                type='CollectGraspNet',
                keys=[
                    'rgb', 'depth', 'origin_depth', 'gt_rect_grasps',
                    'gt_scores', 'gt_object_ids', 'gt_depths'
                ])
        ]),
    val=dict(
        type='GraspNetDataset',
        root='data/planer_graspnet',
        camera='realsense',
        rect_label_folder='rect_labels_filt_top10%_depth2_nms_0.02_10',
        split='test',
        view='1016',
        dump_folder='eval/sim_to_real/rgb_ddd_sim',
        pipeline=[
            dict(
                type='LoadRGBDepthGraspNet',
                with_rgb=True,
                with_depth=True,
                with_origin_depth=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 720),
                flip=False,
                transforms=[
                    dict(
                        type='ResizeGraspNet',
                        with_rgb=True,
                        with_depth=True,
                        with_origin_depth=True,
                        keep_ratio=True),
                    dict(
                        type='NormalizeRGB',
                        mean=[105.4, 118.08, 115.29],
                        std=[71.78, 67.65, 70.59],
                        to_rgb=True),
                    dict(type='NormalizeDepth', mean=477.11, std=101.36),
                    dict(
                        type='PadGraspNet',
                        with_rgb=True,
                        with_depth=True,
                        with_origin_depth=True,
                        size_divisor=32),
                    dict(
                        type='ImageToTensor',
                        keys=['rgb', 'depth', 'origin_depth']),
                    dict(
                        type='CollectGraspNet',
                        keys=['rgb', 'depth', 'origin_depth'])
                ])
        ]),
    test=dict(
        type='GraspNetDataset',
        root='data/planer_graspnet',
        dump_folder='eval/sim_to_real/rgb_ddd_sim',
        camera='realsense',
        rect_label_folder='rect_labels_filt_top10%_depth2_nms_0.02_10',
        split='test',
        view='1016',
        pipeline=[
            dict(
                type='LoadRGBDepthGraspNet',
                with_rgb=True,
                with_depth=True,
                with_origin_depth=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 720),
                flip=False,
                transforms=[
                    dict(
                        type='ResizeGraspNet',
                        with_rgb=True,
                        with_depth=True,
                        with_origin_depth=True,
                        keep_ratio=True),
                    dict(
                        type='NormalizeRGB',
                        mean=[105.4, 118.08, 115.29],
                        std=[71.78, 67.65, 70.59],
                        to_rgb=True),
                    dict(type='NormalizeDepth', mean=477.11, std=101.36),
                    dict(
                        type='PadGraspNet',
                        with_rgb=True,
                        with_depth=True,
                        with_origin_depth=True,
                        size_divisor=32),
                    dict(
                        type='ImageToTensor',
                        keys=['rgb', 'depth', 'origin_depth']),
                    dict(
                        type='CollectGraspNet',
                        keys=['rgb', 'depth', 'origin_depth'])
                ])
        ]))
evaluation = dict(interval=3, metric='grasp')
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=45)
checkpoint_config = dict(interval=3)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/simb2realsense/source_only'
fp16 = dict(loss_scale=512.0)
gpu_ids = range(0, 4)
