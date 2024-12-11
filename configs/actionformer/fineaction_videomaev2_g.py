_base_ = [
    "../_base_/datasets/fineaction/features_internvideo_resize_trunc.py",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

annotation_path = 'data/fineaction/fineaction_60.json'
data_path = "/home1/caisihang/data/fineaction/mae_features_16_16/"
block_list = data_path + "missing_files.txt"
dataset = dict(
    train=dict(data_path=data_path, block_list=block_list),
    val=dict(data_path=data_path, block_list=block_list),
    test=dict(data_path=data_path, block_list=block_list),
)

model = dict(
    projection=dict(
        in_channels=1408,
        out_channels=256,
        attn_cfg=dict(n_mha_win_size=[7, 7, 7, 7, 7, -1]),
        use_abs_pe=True,
        max_seq_len=192,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        num_classes=1,
        label_smoothing=0.1,
        loss_weight=2.0,
        loss_normalizer=200,
    ),
)

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=15)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=100,
        iou_threshold=0,  # does not matter when use soft nms
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.9,  #  set 0 to disable
    ),
    external_cls=dict(
        type="StandardClassifier",
        path="data/fineaction/new_swinB_1x1x256_views2x3_max_label_avg_prob.json",
        topk=2,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=7,
)

work_dir = "exps/fineaction/actionformer_videomaev2_g"
