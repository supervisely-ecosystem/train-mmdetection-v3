import os
from mmengine import Config, ConfigDict
from mmdet.registry import RUNNERS

import supervisely as sly
from supervisely.app.widgets import Card, Button, Container, Progress, Empty

import src.sly_globals as g
from src.train_parameters import TrainParameters
from src.ui.task import task_selector
from src.ui.train_val_split import dump_train_val_splits
from src.ui.classes import classes
import src.ui.models as models_ui
from src import sly_utils
from src.ui.hyperparameters import update_params_with_widgets
from src.ui.augmentations import get_selected_aug
from src.ui.graphics import add_classwise_metric, monitoring

# register modules (don't remove):
from src import sly_dataset, sly_hook, sly_imgaugs


def get_task():
    if "segmentation" in task_selector.get_value().lower():
        return "instance_segmentation"
    else:
        return "object_detection"


def set_device_env(device_name: str):
    if device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device_id = device_name.split(":")[1].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def get_train_params(cfg) -> TrainParameters:
    task = get_task()
    selected_classes = classes.get_selected_classes()
    augs_config_path = get_selected_aug()

    # create params from config
    params = TrainParameters.from_config(cfg)
    params.init(task, selected_classes, augs_config_path, g.app_dir)

    # update params with UI
    update_params_with_widgets(params)
    params.add_classwise_metric = len(selected_classes) <= g.MAX_CLASSES_TO_SHOW_CLASSWISE_METRIC
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


def add_metadata(cfg: Config):
    is_pretrained = models_ui.is_pretrained_model_selected()

    if not is_pretrained and not hasattr(cfg, "sly_metadata"):
        # realy custom model
        sly.logger.warn(
            "There are no sly_metadata in config, seems the custom model wasn't trained in Supervisely."
        )
        cfg.sly_metadata = {
            "model_name": "custom",
            "architecture_name": "custom",
            "task_type": get_task(),
        }

    if is_pretrained:
        selected_model = models_ui.get_selected_pretrained_model()
        metadata = {
            "model_name": selected_model["name"],
            "architecture_name": models_ui.get_selected_arch_name(),
            "task_type": get_task(),
        }
    else:
        metadata = cfg.sly_metadata

    metadata["project_id"] = g.PROJECT_ID
    metadata["project_name"] = g.api.project.get_info_by_id(g.PROJECT_ID).name

    cfg.sly_metadata = ConfigDict(metadata)


def train():
    # download dataset
    project_dir = sly_utils.download_project(iter_progress)

    # prepare split files
    dump_train_val_splits(project_dir)

    # prepare model files
    iter_progress(message="Preparing the model...", total=1)
    config_path, weights_path_or_url = prepare_model()

    # create config
    cfg = Config.fromfile(config_path)
    params = get_train_params(cfg)

    # set device
    # set_device_env(params.device_name)
    # doesn't work :(
    # may because of torch has been imported earlier and it already read CUDA_VISIBLE_DEVICES

    ### TODO: debug
    # params.checkpoint_interval = 5
    # params.save_best = False
    # params.val_interval = 1
    # params.num_workers = 0
    # params.input_size = (409, 640)
    # from mmengine.visualization import Visualizer
    # from mmdet.visualization import DetLocalVisualizer

    # Visualizer._instance_dict.clear()
    # DetLocalVisualizer._instance_dict.clear()
    ###

    # create config from params
    train_cfg = params.update_config(cfg)

    # update load_from with custom_weights_path
    if params.load_from and weights_path_or_url:
        train_cfg.load_from = weights_path_or_url

    # add sly_metadata
    add_metadata(train_cfg)

    # show classwise chart
    if params.add_classwise_metric:
        add_classwise_metric(classes.get_selected_classes())
        sly.logger.debug("Added classwise metrics")

    # update globals
    config_name = config_path.split("/")[-1]
    g.config_name = config_name
    g.params = params

    # clean work_dir
    if sly.fs.dir_exists(params.work_dir):
        sly.fs.remove_dir(params.work_dir)

    # TODO: debug
    # train_cfg.dump("debug_config.py")

    iter_progress(message="Preparing the model...", total=1)

    # Its grace, the Runner!
    runner = RUNNERS.build(train_cfg)
    try:
        runner.train()
    except StopIteration as exc:
        sly.logger.info("The training is stopped.")

    epoch_progress.hide()

    # uploading checkpoints and data
    # TODO: params.experiment_name
    out_path = sly_utils.upload_artifacts(
        params.work_dir,
        params.experiment_name,
        iter_progress,
    )

    # set task results
    if sly.is_production():
        file_id = g.api.file.get_info_by_path(g.TEAM_ID, out_path + "/config.py").id
        g.api.task.set_output_directory(g.api.task_id, file_id, out_path)
        g.app.stop()


start_train_btn = Button("Train")
stop_train_btn = Button("Stop", "danger")
stop_train_btn.disable()

epoch_progress = Progress("Epochs")
epoch_progress.hide()

iter_progress = Progress("Iterations", hide_on_finish=False)
iter_progress.hide()

btn_container = Container(
    [start_train_btn, stop_train_btn, Empty()],
    "horizontal",
    overflow="wrap",
    fractions=[1, 1, 10],
    gap=1,
)

container = Container(
    [
        btn_container,
        epoch_progress,
        iter_progress,
        monitoring.compile_monitoring_container(True),
    ]
)

card = Card(
    "7️⃣ Training progress",
    "Task progress, detailed logs, metrics charts, and other visualizations",
    content=container,
    lock_message="Select a model to unlock.",
)
card.lock()


@start_train_btn.click
def start_train():
    g.stop_training = False
    monitoring.container.show()
    stop_train_btn.enable()
    # epoch_progress.show()
    iter_progress.show()
    train()


@stop_train_btn.click
def stop_train():
    g.stop_training = True
    stop_train_btn.disable()
