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


def download_custom_model(remote_weights_path: str):
    # save_dir structure:
    # - checkpoints
    # - logs
    # - config_xxx.py

    # download .pth
    model_name = remote_weights_path.split("/")[-1]
    weights_path = sly.app.get_data_dir() + f"/{model_name}"
    g.api.file.download(g.TEAM_ID, remote_weights_path, weights_path)

    # download config_xxx.py
    save_dir = remote_weights_path.split("checkpoints")
    files = g.api.file.listdir(g.TEAM_ID, save_dir)
    # find config by name in save_dir
    remote_config_path = [f for f in files if f.endswith(".py")]
    assert len(remote_config_path) > 0, f"Can't find config in {save_dir}."
    remote_config_path = remote_config_path[0]
    config_name = remote_config_path.split("/")[-1]
    config_path = sly.app.get_data_dir() + f"/{config_name}"
    g.api.file.download(g.TEAM_ID, remote_config_path, config_path)

    return weights_path, config_path


def upload_artifacts(experiment_name: str):
    local_dir = sly.app.get_data_dir() + f"/{experiment_name}"
    task_id = g.api.task_id or ""
    g.api.file.upload_directory(g.TEAM_ID, local_dir, f"/mmdetction-2/{experiment_name}_{task_id}")


def prepare_model():
    # download custom model if needed
    # returns config and weights paths
    if models_ui.is_pretrained_model_selected():
        selected_model = models_ui.get_selected_pretrained_model()
        config_path = selected_model["config"]
        custom_weights_path = None
    else:
        remote_weights_path = models_ui.get_selected_custom_path()
        custom_weights_path, config_path = download_custom_model(remote_weights_path)
    return config_path, custom_weights_path


def download_project():
    project_dir = f"{sly.app.get_data_dir()}/sly_project"
    if sly.fs.dir_exists(project_dir):
        sly.fs.remove_dir(project_dir)
    sly.Project.download(g.api, g.PROJECT_ID, project_dir)
    return project_dir


def train():
    # download dataset
    project_dir = download_project()

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
        print(exc)

    upload_artifacts(train_cfg.work_dir)


start_train_btn = Button("Train")
# pause_train_btn = Button("Pause")
stop_train_btn = Button("Stop", "danger")
stop_train_btn.disable()

epoch_progress = Progress("Epoch")
iter_progress = Progress("Iteration")

btn_container = Container([start_train_btn, stop_train_btn], "horizontal", overflow="wrap")
container = Container([btn_container, epoch_progress, iter_progress])


@start_train_btn.click
def start_train():
    start_train_btn.disable()
    stop_train_btn.enable()
    train()


# @pause_train_btn.click
# def pause_train():
#     pass


@stop_train_btn.click
def stop_train():
    # end up training, upload files to Team Files
    g.stop_training = True
