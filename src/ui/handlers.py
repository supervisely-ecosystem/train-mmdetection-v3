import os
import supervisely as sly
from mmengine import Config
from supervisely.app import StateJson

import src.ui.hyperparameters.handlers as handlers

import src.ui.models as models

# from src.ui.utils import button_selected, button_clicked
from src.ui.utils import wrap_button_click, button_clicked
from src.ui.task import select_btn, task_selector
from src.utils import parse_yaml_metafile
from src import sly_utils
from src.train_parameters import TrainParameters
from src.ui import hyperparameters
from src.ui import augmentations
import src.ui.train as train
import src.ui.classes as classes_ui
import src.ui.train_val_split as splits_ui
from src.ui import model_leaderboard


# def model_select_button_state_change(without_click: bool = False):
#     button_selected(
#         models.select_btn,
#         disable_widgets=[
#             models.radio_tabs,
#             models.arch_select,
#             models.path_field,
#             models.table,
#         ],
#         lock_cards=[
#             classes_ui.card,
#             splits_ui.card,
#             augmentations.card,
#             hyperparameters.card,
#             train.card,
#         ],
#         lock_without_click=without_click,
#     )

models_select_callback = wrap_button_click(
    models.select_btn,
    cards_to_unlock=[
        classes_ui.card,
        splits_ui.card,
        augmentations.card,
        hyperparameters.card,
        train.card,
    ],
    widgets_to_disable=[
        models.radio_tabs,
        models.arch_select,
        models.path_field,
        models.table,
        models.load_from,
    ],
    lock_msg="Select a model to unlock.",
)

task_select_callback = wrap_button_click(
    select_btn,
    models.card,
    [task_selector],
    models_select_callback,
    lock_msg="Select a task to unlock.",
)


# TASK
def on_task_changed(selected_task):
    models.update_architecture(selected_task)
    augmentations.update_task(selected_task)
    model_leaderboard.update_table(models.models_meta, selected_task)


@select_btn.click
def select_task():
    task_select_callback()
    if button_clicked[select_btn.widget_id]:
        on_task_changed(task_selector.get_value())
    else:
        model_leaderboard.table.read_json(None)
        model_leaderboard.table.sort(0)


# MODELS
models.update_architecture(task_selector.get_value())


@models.arch_select.value_changed
def on_architecture_selected(selected_arch):
    models.update_models(selected_arch)


@models.table.value_changed
def update_selected_model(selected_row):
    models.update_selected_model(selected_row)


@models.select_btn.click
def on_model_selected():
    # unlock cards
    models_select_callback()

    # update default hyperparameters in UI
    is_pretrained_model = models.is_pretrained_model_selected()

    if is_pretrained_model:
        selected_model = models.get_selected_pretrained_model()
        config_path = selected_model["config"]
    else:
        remote_weights_path = models.get_selected_custom_path()
        assert os.path.splitext(remote_weights_path)[1].startswith(
            ".pt"
        ), "Please, select checkpoint file with model weights (.pth)"
        config_path = sly_utils.download_custom_config(remote_weights_path)

    cfg = Config.fromfile(config_path)
    if not is_pretrained_model:
        # check task type is correct
        model_task = cfg.train_dataloader.dataset.task
        selected_task = train.get_task()
        assert (
            model_task == selected_task
        ), f"The selected model was trained in {model_task} task, but you've selected the {selected_task} task. Please, check your selected task."
        # check if config is from mmdet v3.0
        assert hasattr(
            cfg, "optim_wrapper"
        ), "Missing some parameters in config. Please, check if your custom model was trained in mmdetection v3.0."

    params = TrainParameters.from_config(cfg)
    if params.warmup_iters:
        params.warmup_iters = sly_utils.get_images_count() // 2
    hyperparameters.update_widgets_with_params(params)

    # unlock cards
    sly.logger.debug(f"State {classes_ui.card.widget_id}: {StateJson()[classes_ui.card.widget_id]}")
