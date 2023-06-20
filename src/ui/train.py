from mmengine import Config
from mmdet.registry import RUNNERS

import supervisely as sly
from supervisely.app.widgets import Card, Button

from src.train_parameters import TrainParameters
from src.ui.task import task_selector
from src.ui.train_val_split import dump_train_val_splits
from src.ui.classes import classes
from src.ui.models import table, radio_tabs

# from src.ui.augmentations import augments


def get_task():
    if "segmentation" in task_selector.get_value().lower():
        return "instance_segmentation"
    else:
        return "object_detection"


def get_config() -> Config:
    # это надо перенести в models
    # download custom or get pretrained
    name = table.get_selected_row()[0]
    config_path = "configs/..."
    cfg = Config.fromfile(config_path)
    return cfg


def get_train_params(cfg) -> TrainParameters:
    task = get_task()
    selected_classes = classes.get_selected_classes()
    augs_config_path = ...
    work_dir = sly.app.get_data_dir()

    params = TrainParameters.from_config(cfg)
    params.init(task, selected_classes, augs_config_path, work_dir)
    params.total_epochs = ...
    return params


def train():
    dump_train_val_splits()
    pretrained_cfg = get_config()
    params = get_train_params(pretrained_cfg)
    cfg = params.update_config(pretrained_cfg)
    runner = RUNNERS.build(cfg)
    runner.train()
    # upload checkpoints


start_train_btn = Button("Train")


@start_train_btn.click
def start_train():
    train()
