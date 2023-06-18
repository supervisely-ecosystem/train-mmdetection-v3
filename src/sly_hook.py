from typing import Dict, Optional
from mmdet.registry import HOOKS
from mmengine.hooks import LoggerHook, Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner, LogProcessor
import numpy as np
import supervisely as sly
import torch


@HOOKS.register_module()
class SuperviselyHook(Hook):
    priority = "HIGH"

    def __init__(
        self,
        interval: int = 10,
    ):
        self.interval = interval

    def after_train_iter(
        self, runner: Runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: dict = None
    ) -> None:
        # inputs = torch.cat(data_batch["inputs"], 1).permute(1, 2, 0).cpu().numpy()[..., ::-1]

        # if (inputs.sum(-1) == 0).any():
        #     # sly.image.write(f"inputs_{runner.iter:02}.jpg", inputs)
        #     print()

        if True:
            # sly.image.write("inputs_nan.jpg", inputs)

            # draw with mask
            import cv2

            for b_idx in range(len(data_batch["inputs"])):
                img = data_batch["inputs"][b_idx].permute(1, 2, 0).cpu().numpy()[..., ::-1]
                m = data_batch["data_samples"][b_idx].gt_instances.masks.masks
                m = (m.sum(0).astype(bool) * 255).astype(np.uint8)
                m = m[..., None].repeat(3, -1)
                m = cv2.addWeighted(m, 0.5, img, 0.5, 0)
                for bbox in data_batch["data_samples"][b_idx].gt_instances.bboxes.tensor.to(int):
                    p1 = bbox[:2].tolist()
                    p2 = bbox[2:].tolist()
                    m = cv2.rectangle(
                        m,
                        p1,
                        p2,
                        [
                            255,
                        ],
                        thickness=1,
                    )
                sly.image.write(f"masks_nan_b{b_idx}.jpg", m)
            print()

        # check nans
        if not torch.isfinite(outputs["loss"]):
            print()

        # fill progress
        # ...

        # update plots
        if self.every_n_train_iters(runner, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, "train")
            print(runner.iter, tag)

    def after_val_epoch(self, runner: Runner, metrics: Dict[str, float] = None) -> None:
        classwise_keys = [m for m in metrics if m.endswith("_precision")]
        mAP_keys = ["coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75"]
        dataset_meta = runner.val_dataloader.dataset.metainfo
