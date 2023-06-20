from typing import List
import supervisely as sly
from supervisely.app.widgets import (
    RadioTabs,
    RadioTable,
    Select,
    SelectString,
    Input,
    Card,
    Container,
    Button,
    Text,
    Field,
    TeamFilesSelector,
)

from src.ui.task import task_selector
from src.ui.utils import update_custom_params
from src.sly_globals import TEAM_ID
from src.utils import parse_yaml_metafile


def load_models_meta(task: str):
    if "segmentation" in task.lower():
        models_meta = sly.json.load_json_file("models/instance_segmentation_meta.json")
    else:
        models_meta = sly.json.load_json_file("models/detection_meta.json")
    models_meta = {m["model_name"]: m for m in models_meta}
    return models_meta


def get_architecture_list(models_meta: dict):
    arch_names = list(models_meta.keys())

    labels = []
    right_texts = []
    for name, item in models_meta.items():
        if item.get("paper_from") and item.get("year"):
            label = f"{name}"
            r_text = f"({item.get('paper_from')} {item.get('year')})"
        else:
            label = f"{name}"
            r_text = ""
        labels.append(label)
        right_texts.append(r_text)

    # links to README.md in mmdetection repo
    base_url = "https://github.com/open-mmlab/mmdetection/tree/main/configs/"
    links = [base_url + m["yml_file"].split("/")[0] for m in models_meta.values()]

    return arch_names, labels, right_texts, links


def get_models_by_architecture(task: str, models_meta: dict, selected_arch_name: str):
    # parse metafile.yml
    metafile_path = "configs/" + models_meta[selected_arch_name]["yml_file"]
    _, models = parse_yaml_metafile(metafile_path)

    # filter models by task
    if "segmentation" in task.lower():
        task_name = "Instance Segmentation"
    else:
        task_name = "Object Detection"
    models = [m for m in models if task_name in m["tasks"]]
    return models


def get_table_data(task: str, models: list):
    columns = [
        "Name",
        "Method",
        "Dataset",
        "Inference Time (ms/im)",
        "Training Memory (GB)",
        "box AP",
    ]
    keys = [
        "name",
        "method",
        "dataset",
        "inference_time",
        "train_memory",
        "box AP",
    ]
    if "segmentation" in task.lower():
        columns.append("mask AP")
        keys.append("mask AP")

    # check which keys are used
    add_train_iters = False
    add_train_epochs = False
    for model in models:
        if not add_train_iters and model.get("train_iters"):
            add_train_iters = True
            keys.insert(4, "train_iters")
            columns.insert(4, "Training Iterations")
        if not add_train_epochs and model.get("train_epochs"):
            add_train_epochs = True
            keys.insert(4, "train_epochs")
            columns.insert(4, "Training Epochs")

    # collect rows
    rows = []
    for model in models:
        row = [model.get(k, "-") for k in keys]
        rows.append(row)
    return columns, rows


cur_task = task_selector.get_value()
models_meta = load_models_meta(cur_task)
arch_names, labels, right_texts, links = get_architecture_list(models_meta)

arch_select = SelectString(
    arch_names,
    labels,
    items_right_text=right_texts,
    items_links=links,
)
models = get_models_by_architecture(cur_task, models_meta, arch_names[0])
columns, rows = get_table_data(cur_task, models)
table = RadioTable(columns, rows)

text = Text(text=f"selected model: {table.get_selected_row()[0]}")

# input_file = Input(placeholder="Path to .pth file in Team Files")
input_file = TeamFilesSelector(TEAM_ID, selection_file_type="file")
path_field = Field(
    title="Path to weights file",
    description="Copy path in Team Files",
    content=input_file,
)

radio_tabs = RadioTabs(
    titles=["Pretrained models", "Custom weights"],
    contents=[
        Container(widgets=[arch_select, table, text]),
        path_field,
    ],
)

select_btn = Button(text="Select model")

card = Card(
    title=f"2️⃣{cur_task} models",
    description="Choose model architecture and how weights should be initialized",
    content=Container([radio_tabs, select_btn]),
)


@task_selector.value_changed
def update_architecture(selected_task):
    global models_meta, cur_task
    cur_task = selected_task
    models_meta = load_models_meta(selected_task)
    arch_names, labels, right_texts, links = get_architecture_list(models_meta)
    arch_select.set(arch_names, labels, right_texts, links)
    update_custom_params(card, {"title": f"3️⃣{selected_task} models"})


@arch_select.value_changed
def update_models(selected_arch):
    global models_meta, cur_task
    models = get_models_by_architecture(cur_task, models_meta, selected_arch)
    columns, rows = get_table_data(cur_task, models)
    subtitles = [None] * len(columns)
    table.set_data(columns, rows, subtitles)
    table.select_row(0)
    text.text = f"selected model: {table.get_selected_row()[0]}"


@table.value_changed
def update_selected_model(selected_row):
    text.text = f"selected model: {selected_row[0]}"
