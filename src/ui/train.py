from mmengine import Config
from mmdet.registry import RUNNERS

import supervisely as sly
from supervisely.app.widgets import Card, Button, Container, Progress

import src.sly_globals as g
from src.train_parameters import TrainParameters
from src.ui.task import task_selector
from src.ui.train_val_split import dump_train_val_splits
from src.ui.classes import classes
import src.ui.models as models_ui
from src import sly_utils

# from src.ui.augmentations import augments

# register modules (don't remove):
from src import sly_dataset, sly_hook, sly_imgaugs


def get_task():
    if "segmentation" in task_selector.get_value().lower():
        return "instance_segmentation"
    else:
        return "object_detection"


def get_train_params(cfg) -> TrainParameters:
    task = get_task()
    selected_classes = classes.get_selected_classes()
    # augs_config_path = ...
    augs_config_path = "src/aug_templates/medium.json"
    work_dir = sly.app.get_data_dir()

    params = TrainParameters.from_config(cfg)
    params.init(task, selected_classes, augs_config_path, work_dir)
    # params.total_epochs = ...
    return params


def prepare_model():
    # download custom model if needed
    # returns config and weights paths
    if models_ui.is_pretrained_model_selected():
        selected_model = models_ui.get_selected_pretrained_model()
        config_path = selected_model["config"]
        custom_weights_path = None
    else:
        remote_weights_path = models_ui.get_selected_custom_path()
        custom_weights_path, config_path = sly_utils.download_custom_model(remote_weights_path)
    return config_path, custom_weights_path


def train():
    # download dataset
    project_dir = sly_utils.download_project()

    # prepare split files
    dump_train_val_splits(project_dir)

    # prepare model files
    config_path, custom_weights_path = prepare_model()

    # create config
    cfg = Config.fromfile(config_path)
    params = get_train_params(cfg)
    # update config with user's parameteres
    train_cfg = params.update_config(cfg)
    # update load_from with custom_weights_path
    if custom_weights_path and params.load_from:
        train_cfg.load_from = custom_weights_path

    # dump config locally
    config_name = config_path.split("/")[-1]
    train_cfg.dump(f"{sly.app.get_data_dir()}/{config_name}")

    # its grace, the Trainer
    runner = RUNNERS.build(train_cfg)
    try:
        runner.train()
    except StopIteration as exc:
        # stop training
        print(exc)

    # sly_utils.upload_artifacts(train_cfg.work_dir)


start_train_btn = Button("Train")
# pause_train_btn = Button("Pause")
stop_train_btn = Button("Stop", "danger")
stop_train_btn.disable()

epoch_progress = Progress("Epoch")
iter_progress = Progress("Iteration")

btn_container = Container([start_train_btn, stop_train_btn], "horizontal", overflow="scroll")
container = Container([btn_container, epoch_progress, iter_progress])
# card = Card("Training progress", content=container)


@start_train_btn.click
def start_train():
    stop_train_btn.enable()
    train()


# @pause_train_btn.click
# def pause_train():
#     pass


@stop_train_btn.click
def stop_train():
    # end up training, upload files to Team Files
    g.stop_training = True
