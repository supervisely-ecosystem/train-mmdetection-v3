import src.ui.hyperparameters.handlers as handlers

from src.ui.utils import update_custom_button_params, update_custom_params
from src.ui.task import select_btn, task_selector, reselect_params, select_params
from src.ui.models import (
    table as models_table,
    text as models_desc,
    get_table_data,
    get_models_by_architecture,
    get_architecture_list,
    load_models_meta,
    arch_select,
    card as models_card,
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
@task_selector.value_changed
def update_architecture(selected_task):
    global models_meta, cur_task
    cur_task = selected_task
    models_meta = load_models_meta(selected_task)
    arch_names, labels, right_texts, links = get_architecture_list(models_meta)
    arch_select.set(arch_names, labels, right_texts, links)
    update_custom_params(models_card, {"title": f"2️⃣{selected_task} models"})


@arch_select.value_changed
def update_models(selected_arch):
    global models_meta, cur_task
    models = get_models_by_architecture(cur_task, models_meta, selected_arch)
    columns, rows = get_table_data(cur_task, models)
    subtitles = [None] * len(columns)
    models_table.set_data(columns, rows, subtitles)
    models_table.select_row(0)
    models_desc.text = f"selected model: {models_table.get_selected_row()[0]}"


@models_table.value_changed
def update_selected_model(selected_row):
    models_desc.text = f"selected model: {selected_row[0]}"
