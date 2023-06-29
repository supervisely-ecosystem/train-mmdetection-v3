import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
from mmengine import Config
from mmengine.runner import Runner
from mmdet import registry
import yaml
from src.train_parameters import TrainParameters
from src.utils import parse_yaml_metafile

# register modules (don't remove):
from src import sly_dataset, sly_imgaugs


def json_dump(obj, file):
    with open(file, "w") as f:
        json.dump(obj, f, indent=2)


def json_load(file):
    with open(file, "r") as f:
        return json.load(f)


def get_config_path(model_meta: dict):
    metafile_path = "configs/" + model_meta["yml_file"]
    exclude = model_meta.get("exclude")
    _, models = parse_yaml_metafile(metafile_path, exclude)
    model_item = models[0]
    return model_item["config"]


def run_test(config_path, task):
    cfg = Config.fromfile(config_path)

    selected_classes = ["kiwi", "lemon"]
    augs_config_path = "src/test/medium_test.json"

    # create config
    cfg = Config.fromfile(config_path)

    params = TrainParameters.from_config(cfg)
    params.init(task, selected_classes, augs_config_path, app_dir="app_data")

    params.total_epochs = 2
    params.checkpoint_interval = 6
    params.save_best = False
    params.save_last = False
    params.val_interval = 2
    params.num_workers = 2
    params.input_size = (409, 640)
    # from mmengine.visualization import Visualizer
    # from mmdet.visualization import DetLocalVisualizer

    # Visualizer._instance_dict.clear()
    # DetLocalVisualizer._instance_dict.clear()

    # create config from params
    train_cfg = params.update_config(cfg)

    train_cfg.default_hooks.visualization = dict(
        type="DetVisualizationHook", draw=True, interval=12
    )
    train_cfg.custom_hooks.pop(-1)
    train_cfg.load_from = None
    train_cfg.log_level = "ERROR"

    # cfg.randomness = dict(seed=875212355, deterministic=False)
    runner: Runner = registry.RUNNERS.build(train_cfg)
    runner.train()


det_meta = json_load("models/detection_meta.json")
segm_meta = json_load("models/instance_segmentation_meta.json")

for task, models in zip(TrainParameters.ACCEPTABLE_TASKS, [det_meta, segm_meta]):
    print("TASK:", task)
    for model_meta in models[:5]:
        config_path = get_config_path(model_meta)
        print(f"Selected config: {config_path}")
        run_test(config_path, task)
