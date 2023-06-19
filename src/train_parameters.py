from typing import List
from mmengine import Config, ConfigDict
import multiprocessing
from copy import deepcopy

# register modules:
from src import sly_dataset, sly_hook, sly_imgaugs


class TrainParameters:
    ACCEPTABLE_TASKS = ["instance_segmentation", "detection"]

    def __init__(self) -> None:
        self.task = None
        self.num_classes = None
        self.augs_config_path = None
        self.epoch_based_train = True
        self.total_epochs = 20
        self.val_interval = 1 if self.epoch_based_train else 1000
        self.batch_size_train = 2
        self.batch_size_val = 1
        self.num_workers = min(4, multiprocessing.cpu_count())
        self.checkpoint_interval = 1
        self.optimizer = dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
        self.clip_grad_norm = None
        self.warmup_steps = 0
        self.scheduler = None
        self.losses = None
        self.log_interval = 50
        self.chart_update_interval = 5
        self.load_checkpoint = None  # path or http link

    @classmethod
    def from_config(cls, config):
        cfg = config
        self = cls()
        # TODO: load from config:
        # self.optim_wrapper = None
        self.scheduler = None
        self.losses = None
        return self

    def init(self, task, num_classes, augs_config_path):
        self.task = task
        self.num_classes = num_classes
        self.augs_config_path = augs_config_path

    def update_config(self, config: Config):
        cfg = deepcopy(config)
        assert self.is_inited(), "Please, first call self.init(...) to fill required parameters."

        # change model num_classes
        modify_num_classes_recursive(cfg.model, self.num_classes)

        # pipelines
        train_pipeline, test_pipeline = get_default_pipelines()
        img_aug = dict(type="SlyImgAugs", config_path=self.augs_config_path)
        idx_insert = find_index_for_imgaug(train_pipeline)  # 2 by default
        train_pipeline.insert(idx_insert, img_aug)
        # remove unused:
        if cfg.get("train_pipeline"):
            cfg.train_pipeline = None
        if cfg.get("test_pipeline"):
            cfg.test_pipeline = None

        # datasets
        train_dataset = dict(
            type="SuperviselyDatasetSplit",
            data_root="sly_project",
            split_file="train_split.json",
            pipeline=train_pipeline,
        )
        val_dataset = dict(
            type="SuperviselyDatasetSplit",
            data_root="sly_project",
            split_file="val_split.json",
            save_coco_ann_file="val_coco_instances.json",
            pipeline=test_pipeline,
            test_mode=True,
        )

        # dataloaders
        train_dataloader, val_dataloader = get_default_dataloaders()

        train_dataloader.batch_size = self.batch_size_train
        train_dataloader.num_workers = self.num_workers
        train_dataloader.persistent_workers = self.num_workers != 0
        val_dataloader.batch_size = self.batch_size_val
        val_dataloader.num_workers = self.num_workers
        val_dataloader.persistent_workers = self.num_workers != 0

        train_dataloader.dataset = train_dataset
        val_dataloader.dataset = val_dataset

        cfg.train_dataloader = train_dataloader
        cfg.val_dataloader = val_dataloader
        cfg.test_dataloader = cfg.val_dataloader.copy()

        # evaluators
        # from mmdet.evaluation.metrics import CocoMetric
        coco_metric = "segm" if self.task == "instance_segmentation" else "bbox"
        classwise = self.num_classes <= 10

        cfg.val_evaluator = dict(
            type="CocoMetric",
            ann_file="val_coco_instances.json",
            metric=coco_metric,
            classwise=classwise,
        )

        cfg.test_evaluator = cfg.val_evaluator.copy()

        # train/val
        cfg.train_cfg = dict(
            by_epoch=self.epoch_based_train,
            max_epochs=self.total_epochs,
            val_interval=self.val_interval,
        )

        # hooks
        # from sly_hook import SuperviselyHook
        # from mmdet.engine.hooks import CheckInvalidLossHook, MeanTeacherHook, NumClassCheckHook
        # from mmengine.hooks import CheckpointHook
        cfg.default_hooks.checkpoint = dict(
            type="CheckpointHook",
            interval=self.checkpoint_interval,
            by_epoch=self.epoch_based_train,
        )
        cfg.log_processor = dict(type="LogProcessor", window_size=10, by_epoch=True)
        cfg.default_hooks.logger["interval"] = self.log_interval
        cfg.custom_hooks = [
            dict(type="NumClassCheckHook"),
            dict(type="CheckInvalidLossHook", interval=1),
            dict(type="SuperviselyHook", interval=self.chart_update_interval),
        ]

        # visualization
        cfg.default_hooks.visualization = dict(type="DetVisualizationHook", draw=True, interval=12)

        # optimizer
        # from mmengine.optim.optimizer import OptimWrapper
        cfg.optim_wrapper.optimizer = self.optimizer
        if self.clip_grad_norm:
            cfg.optim_wrapper.clip_grad = dict(max_norm=self.clip_grad_norm)

        # scheduler
        # from mmengine.optim.scheduler import ConstantLR, LinearLR
        cfg.param_scheduler = []
        if self.warmup_steps:
            warmup = dict(
                type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=self.warmup_steps
            )
            cfg.param_scheduler.append(warmup)
        if self.scheduler:
            if self.scheduler["by_epoch"] is False:
                self.scheduler["begin"] = self.warmup_steps
            cfg.param_scheduler.append(self.scheduler)

        # losses
        # TODO
        # can we correctly change losses?

        cfg.work_dir = "work_dirs"
        cfg.load_from = self.load_checkpoint

        return cfg

    def is_inited(self):
        need_to_check = [self.task, self.num_classes, self.augs_config_path]
        return all([bool(x) for x in need_to_check]) and self.task in self.ACCEPTABLE_TASKS


def modify_num_classes_recursive(d, num_classes):
    if isinstance(d, ConfigDict):
        if d.get("num_classes") is not None:
            d["num_classes"] = num_classes
        for k, v in d.items():
            modify_num_classes_recursive(v, num_classes)
    elif isinstance(d, (list, tuple)):
        for v in d:
            modify_num_classes_recursive(v, num_classes)


def find_index_for_imgaug(pipeline):
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


def get_default_pipelines():
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        # *imgagus will be here
        dict(type="Resize", scale=(1333, 800), keep_ratio=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(1333, 800), keep_ratio=True),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    return train_pipeline, test_pipeline


def get_default_dataloaders():
    train_dataloader = dict(
        batch_size=2,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type="DefaultSampler", shuffle=True),
        batch_sampler=dict(type="AspectRatioBatchSampler"),
        dataset=None,
    )

    val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=None,
    )

    return ConfigDict(train_dataloader), ConfigDict(val_dataloader)
