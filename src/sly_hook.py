from typing import Dict, Optional
from mmdet.registry import HOOKS
from mmengine.hooks import LoggerHook, Hook, CheckpointHook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner, LogProcessor
import numpy as np
import supervisely as sly
import torch

import src.sly_globals as g
import src.ui.train as train_ui


@HOOKS.register_module()
class SuperviselyHook(Hook):
    priority = "LOW"

    def __init__(
        self,
        chart_update_interval: int = 5,
    ):
        self.chart_update_interval = chart_update_interval
        self.epoch_progress = None
        self.iter_progress = None

    def before_train(self, runner: Runner) -> None:
        self.epoch_progress = train_ui.epoch_progress(total=runner.max_epochs)
        self.iter_progress = train_ui.iter_progress(total=len(runner.train_dataloader))

    def after_train_iter(
        self, runner: Runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: dict = None
    ) -> None:
        # check nans
        if not torch.isfinite(outputs["loss"]):
            sly.logger.warn("The loss is NaN.")

        # fill progress
        self.iter_progress.update(1)
        # print(runner.epoch, runner.max_epochs, batch_idx+1, len(runner.train_dataloader))

        # update train charts
        if self.every_n_train_iters(runner, self.chart_update_interval):
            tag, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, "train")
            # print(runner.iter, tag)

        # STOP TRAINING
        if g.stop_training:
            sly.logger.info("The training is stopped by user.")
            # force upload (maybe not here)
            raise StopIteration()

    def after_val_epoch(self, runner: Runner, metrics: Dict[str, float] = None) -> None:
        # update val charts
        classwise_keys = [m for m in metrics if m.endswith("_precision")]
        mAP_keys = ["coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75"]
        dataset_meta = runner.val_dataloader.dataset.metainfo
