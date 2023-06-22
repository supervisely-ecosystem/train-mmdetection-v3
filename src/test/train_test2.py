import json
from mmengine import Config
from mmdet import registry
import yaml
from src.train_parameters import TrainParameters

# register modules (don't remove):
from src import sly_dataset, sly_hook, sly_imgaugs


def parse_yaml_metafile(yaml_file):
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    collections = {}  # Name: metadata
    yaml_models = []
    if isinstance(yaml_content, dict):
        if yaml_content.get("Collections"):
            if isinstance(yaml_content["Collections"], list):
                for c in yaml_content["Collections"]:
                    collections[c["Name"]] = c
            else:
                raise NotImplementedError()
        else:
            print(f"Has not collections: {yaml_file}.")
        if yaml_content.get("Models"):
            yaml_models = yaml_content["Models"]
    elif isinstance(yaml_content, list):
        yaml_models = yaml_content
        print(f"Only list: {yaml_file}.")
    else:
        raise NotImplementedError()

    models = []
    for m in yaml_models:
        if not m.get("Weights"):
            print(f"skip {m['Name']} in {yaml_file}, weights don't exists.")
            continue
        # collection = m["In Collection"]
        metrics = {}
        for result in m["Results"]:
            for metric_name, metric_val in result["Metrics"].items():
                metrics[metric_name] = metric_val
            metrics["dataset"] = result["Dataset"]
        m = {
            "name": m["Name"],
            "config": m["Config"],
            "tasks": [r["Task"] for r in m["Results"]],
            "weights": m["Weights"],
            **metrics,
        }
        models.append(m)

    return collections, models


def json_dump(obj, file):
    with open(file, "w") as f:
        json.dump(obj, f, indent=2)


def json_load(file):
    with open(file, "r") as f:
        return json.load(f)


def get_manual_config_path():
    det_meta = json_load("models/detection_meta.json")
    segm_meta = json_load("models/instance_segmentation_meta.json")

    # select model
    model_item = segm_meta[0]
    _, models = parse_yaml_metafile("configs/" + model_item["yml_file"])
    model_item = models[0]
    return model_item["config"]


config_path = (
    # "tmp_cfg.py"
    # "configs/convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py"
    # "configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"
    # "configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py"
    get_manual_config_path()
)
print(f"Selected config: {config_path}")

cfg = Config.fromfile(config_path)

task = "instance_segmentation"
selected_classes = ["kiwi"]
augs_config_path = "src/aug_templates/medium.json"

params = TrainParameters.from_config(cfg)
params.init(task, selected_classes, augs_config_path, "work_dirs")
params.batch_size_train = 2
params.checkpoint_interval = 15
params.val_interval = 3
params.batch_size_val = 1
params.num_workers = 2
params.total_epochs = 15

cfg = params.update_config(cfg)
cfg.custom_hooks = []

# TODO: to_del:
# cfg.optim_wrapper = dict(
#     type="OptimWrapper",
#     optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.0001),
#     paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1, decay_mult=1.0)}),
#     clip_grad=dict(max_norm=1.0, norm_type=2),
# )
cfg.load_from = "https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth"
# cfg.resume = False


# cfg.randomness = dict(seed=875212355, deterministic=False)
runner = registry.RUNNERS.build(cfg)
runner.train()

from mmengine.runner import Runner
