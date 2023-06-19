from mmengine import Config
from mmdet import registry
from src.train_parameters import TrainParameters


config_path = (
    # "tmp_cfg.py"
    # "configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py"
    "configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"
    # "configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py"
)

cfg = Config.fromfile(config_path)

task = "instance_segmentation"
num_classes = 2
augs_config_path = "medium.json"

params = TrainParameters.from_config(cfg)
params.init(task, num_classes, augs_config_path)
params.batch_size_train = 2
params.checkpoint_interval = 15
params.val_interval = 15
params.batch_size_val = 1
params.num_workers = 0

cfg = params.update_config(cfg)

# cfg.randomness = dict(seed=875212355, deterministic=False)
runner = registry.RUNNERS.build(cfg)
runner.train()

from mmengine.runner import Runner
