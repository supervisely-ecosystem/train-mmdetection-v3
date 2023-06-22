import src.ui.hyperparameters.handlers as handlers

from src.ui.utils import update_custom_button_params, update_custom_params
from src.ui.task import select_btn, task_selector, reselect_params, select_params
from src.ui.models import (
    table as models_table,
    text as models_desc,
    update_models,
    update_architecture,
    update_selected_model,
    arch_select,
)
from src.utils import parse_yaml_metafile


# TASK
@select_btn.click
def select_task():
    if select_btn._click_handled:
        task_selector.disable()
        update_custom_button_params(select_btn, reselect_params)
        select_btn._click_handled = False
        # TODO: load task config if selected
    else:
        task_selector.enable()
        update_custom_button_params(select_btn, select_params)
        select_btn._click_handled = True
        # TODO: restart all steps


# MODELS
update_architecture(task_selector.get_value())


@task_selector.value_changed
def on_task_changed(selected_task):
    update_architecture(selected_task)


@arch_select.value_changed
def on_architecture_selected(selected_arch):
    update_models(selected_arch)


@models_table.value_changed
def update_selected_model(selected_row):
    models_desc.text = f"Selected model: {selected_row[0]}"
