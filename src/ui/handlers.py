from mmengine import Config

import src.ui.hyperparameters.handlers as handlers

import src.ui.models as models
from src.ui.utils import button_selected
from src.ui.task import select_btn, task_selector
from src.utils import parse_yaml_metafile
from src import sly_utils
from src.train_parameters import TrainParameters
from src.ui import hyperparameters
from src.ui import augmentations


# TASK
def on_task_changed(selected_task):
    models.update_architecture(selected_task)
    augmentations.update_task(selected_task)


@select_btn.click
def select_task():
    button_selected(select_btn, [task_selector], [models.card])
    on_task_changed(task_selector.get_value())


# MODELS
models.update_architecture(task_selector.get_value())


@models.arch_select.value_changed
def on_architecture_selected(selected_arch):
    models.update_models(selected_arch)


@models.table.value_changed
def update_selected_model(selected_row):
    models.text.text = f"Selected model: {selected_row[0]}"


@models.select_btn.click
def on_model_selected():
    # update default hyperparameters in UI
    button_selected(
        models.select_btn,
        disable_widgets=[
            models.radio_tabs,
            models.arch_select,
            models.path_field,
            models.table,
        ],
        unlock_cards=[augmentations.card, hyperparameters.card],
    )

    if models.is_pretrained_model_selected():
        selected_model = models.get_selected_pretrained_model()
        config_path = selected_model["config"]
    else:
        remote_weights_path = models.get_selected_custom_path()
        config_path = sly_utils.download_custom_config(remote_weights_path)
    cfg = Config.fromfile(config_path)
    params = TrainParameters.from_config(cfg)
    hyperparameters.update_widgets_with_params(params)

    # unlock card
    hyperparameters.card.unlock()
