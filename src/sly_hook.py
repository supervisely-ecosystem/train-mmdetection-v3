from typing import Dict, Optional
from mmdet.registry import HOOKS
from mmengine.hooks import LoggerHook, Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner, LogProcessor
import numpy as np
import supervisely as sly


@HOOKS.register_module()
class SuperviselyHook(Hook):
    priority = "LOW"

    def __init__(
        self,
        interval: int = 10,
    ):
        self.interval = interval

    def after_train_iter(
        self, runner: Runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: dict = None
    ) -> None:
        # fill progress

        # update plots
        if self.every_n_train_iters(runner, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, "train")
            print(runner.iter, tag)

    def after_val_epoch(self, runner: Runner, metrics: Dict[str, float] = None) -> None:
        classwise_keys = [m for m in metrics if m.endswith("_precision")]
        mAP_keys = ["coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75"]
        dataset_meta = runner.val_dataloader.dataset.metainfo
