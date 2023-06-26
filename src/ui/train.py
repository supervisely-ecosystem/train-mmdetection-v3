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
from src.ui.hyperparameters import update_params_with_widgets
from src.ui.augmentations import get_selected_aug
from src.ui.graphics import add_classwise_metric

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
    augs_config_path = get_selected_aug()

    # create params from config
    params = TrainParameters.from_config(cfg)
    params.init(task, selected_classes, augs_config_path, g.app_dir)

    # update params with UI
    update_params_with_widgets(params)
    params.load_from = models_ui.load_from.is_switched()
    params.add_classwise_metric = len(selected_classes) <= g.MAX_CLASSES_TO_SHOW_CLASSWISE_METRIC
    # TODO: filter_images_without_gt
    # params.filter_images_without_gt = ...
    return params


def prepare_model():
    # download custom model if needed
    # returns config path and weights path
    if models_ui.is_pretrained_model_selected():
        selected_model = models_ui.get_selected_pretrained_model()
        config_path = selected_model["config"]
        weights_path_or_url = selected_model["weights"]
    else:
        remote_weights_path = models_ui.get_selected_custom_path()
        weights_path_or_url, config_path = sly_utils.download_custom_model(remote_weights_path)
    return config_path, weights_path_or_url


def train():
    # download dataset
    project_dir = sly_utils.download_project()

    # prepare split files
    dump_train_val_splits(project_dir)

    # prepare model files
    config_path, weights_path_or_url = prepare_model()

    # create config
    cfg = Config.fromfile(config_path)
    params = get_train_params(cfg)

    ### TODO: debug
    params.checkpoint_interval = 5
    params.save_best = False
    params.val_interval = 1
    params.num_workers = 0
    ###

    # get config from params
    train_cfg = params.update_config(cfg)
    # update load_from with custom_weights_path
    if params.load_from and weights_path_or_url:
        train_cfg.load_from = weights_path_or_url

    if params.add_classwise_metric:
        add_classwise_metric(classes.get_selected_classes())
        sly.logger.debug("Added classwise metrics")

    # add in globals
    config_name = config_path.split("/")[-1]
    g.config_name = config_name
    g.params = params

    # clean work_dir
    if sly.fs.dir_exists(params.work_dir):
        sly.fs.remove_dir(params.work_dir)

    # its grace, the Trainer
    runner = RUNNERS.build(train_cfg)
    try:
        runner.train()
    except StopIteration as exc:
        # training was stopped
        sly.logger.debug(exc)

    # TODO: params.experiment_name
    sly_utils.upload_artifacts(
        params.work_dir,
        params.experiment_name,
        # TODO: make correct monitor
        iter_progress(message="Uploading to Team Files...", unit="MB"),
    )


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
    stop_train_btn.disable()
