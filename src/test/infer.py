import mmcv
from mmengine import Config
from mmengine.runner import load_checkpoint
from mmengine.model.utils import revert_sync_batchnorm
from mmdet.apis import inference_detector, init_detector
import sys
from src.sly_dataset import SuperviselyDatasetSplit

import mmengine.config

sys.modules["mmcv.utils.config"] = mmengine.config


# img_path = "sly_project/ds1/img/IMG_0748.jpeg"
# config_path = "work_dirs/20230614_060327.py"
# weights_path = "work_dirs/epoch_5.pth"

img_path = "demo_data/image_01.jpg"
config_path = "configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"
weights_path = "cascade_rcnn_1x.pth"

device = "cuda:0"


cfg = Config.fromfile(config_path)
model = init_detector(cfg, weights_path, device=device)
img = mmcv.imread(img_path, channel_order="rgb")
result = inference_detector(model, img)
print(result)

from mmdet.registry import VISUALIZERS
from mmdet.visualization.local_visualizer import DetLocalVisualizer

cfg.vis_backends[0]["save_dir"] = "vis"
model.cfg.visualizer["save_dir"] = "vis"
visualizer: DetLocalVisualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
visualizer.add_datasample(
    "result",
    img,
    data_sample=result,
    draw_gt=False,
    wait_time=0,
    show=False,
)
# img_d = visualizer.get_image()
# visualizer.add_image("test.png", img_d)

# in config file add:
# custom_imports = dict(imports=['my_module'], allow_failed_imports=False)
