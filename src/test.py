from mmengine import Config, ConfigDict
from mmdet import registry

from sly_dataset import SuperviselyDatasetSplit
from sly_imgaugs import SlyImgAugs

# from sly_coco_metric import SuperviselyCocoMetric

config_path = (
    # "tmp_cfg.py"
    # "configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py"
    "configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"
    # "configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py"
)

cfg = Config.fromfile(config_path)


def modify_num_classes_recursive(d, num_classes):
    if isinstance(d, ConfigDict):
        if d.get("num_classes") is not None:
            d["num_classes"] = num_classes
        for k, v in d.items():
            modify_num_classes_recursive(v, num_classes)
    elif isinstance(d, (list, tuple)):
        for v in d:
            modify_num_classes_recursive(v, num_classes)


def find_index_for_aug(pipeline):
    # return index after LoadImageFromFile and LoadAnnotations
    i1, i2 = -1, -1
    types = [p["type"] for p in pipeline]
    if "LoadImageFromFile" in types:
        i1 = types.index("LoadImageFromFile")
    if "LoadAnnotations" in types:
        i2 = types.index("LoadAnnotations")
    idx_insert = max(i1, i2)
    if idx_insert != -1:
        idx_insert += 1
    return idx_insert


# change num_classes
modify_num_classes_recursive(cfg.model, 2)


# pipelines
augs_config = "medium.json"
img_aug = dict(type="SlyImgAugs", config_path=augs_config)
idx_insert = find_index_for_aug(cfg.train_pipeline)
# cfg.train_pipeline.insert(idx_insert, img_aug)


# datasets
cfg.train_dataloader.dataset = dict(
    type="SuperviselyDatasetSplit",
    data_root="sly_project",
    split_file="train_split.json",
    pipeline=cfg.train_pipeline,
)

cfg.val_dataloader.dataset = dict(
    type="SuperviselyDatasetSplit",
    data_root="sly_project",
    split_file="val_split.json",
    save_coco_ann_file="val_coco_instances.json",
    pipeline=cfg.test_pipeline,
    test_mode=True,
)

cfg.test_dataloader = cfg.val_dataloader.copy()

# evaluators
cfg.val_evaluator = dict(
    type="CocoMetric",
    ann_file="val_coco_instances.json",
    metric=["bbox", "segm"],
    format_only=False,
)

cfg.test_evaluator = cfg.val_evaluator.copy()

# hooks, vis
vis_hook = cfg.default_hooks.visualization
vis_hook["draw"] = True
# vis_hook["test_out_dir"] = "vis_draw_dir"
vis_hook["interval"] = 12

cfg.default_hooks.checkpoint["interval"] = 12
cfg.default_hooks.logger["interval"] = 20

# optimizer
opt = dict(type="AdamW", lr=0.0002, betas=(0.9, 0.999), weight_decay=0.0005)
cfg.optim_wrapper.optimizer = opt
cfg.optim_wrapper.clip_grad = dict(max_norm=20.0)
cfg.param_scheduler = []

# train/val
cfg.train_cfg["val_interval"] = 12

from mmengine.optim.optimizer import OptimWrapper

# from mmengine.optim.scheduler import ConstantLR


# run
cfg.work_dir = "work_dirs"
runner = registry.RUNNERS.build(cfg)
runner.train()
