from typing import List
from supervisely.app.widgets import (
    RadioTabs,
    RadioTable,
    Select,
    Input,
    Card,
    Container,
    Button,
    Text,
    Field,
)

from src.ui.task import task_selector


def get_architectures_by_task(task: str) -> List[Select.Item]:
    if "segmentation" in task:
        s = 4
    else:
        s = 0

    archs = [Select.Item(value=f"{i}", label=f"l{i}") for i in range(s, s + 4)]
    arch_links = [
        "https://github.com/open-mmlab/mmdetection/tree/v2.22.0/configs/queryinst"
        for _ in range(len(archs))
    ]
    return archs, arch_links


def get_table_columns(metrics):
    columns = [
        {"key": "name", "title": "Checkpoint", "subtitle": None},
        {"key": "method", "title": "Method", "subtitle": None},
        {"key": "dataset", "title": "Dataset", "subtitle": None},
        {"key": "inference_time", "title": "Inference time", "subtitle": "(ms/im)"},
        {"key": "resolution", "title": "Input size", "subtitle": "(H, W)"},
        {"key": "epochs", "title": "Epochs", "subtitle": None},
        {"key": "training_memory", "title": "Memory", "subtitle": "Training (GB)"},
    ]
    for metric in metrics:
        columns.append({"key": metric, "title": metric, "subtitle": "score"})

    titles = [c["title"] for c in columns]
    subtitles = [c["subtitle"] for c in columns]
    return titles, subtitles, columns


def get_models_by_architecture(titles, architecture):
    # models class
    ml = int(architecture) + 2
    return [list(range(i, i + len(titles))) for i in range(ml)]


arch, links = get_architectures_by_task(task_selector.get_value())

arch_select = Select(
    items=arch,
    items_links=links,
)

titles, subtitles, _ = get_table_columns(["ROI"])
table = RadioTable(
    columns=titles,
    rows=get_models_by_architecture(titles, arch_select.get_value()),
    subtitles=subtitles,
)

text = Text(text=f"selected model: {table.get_selected_row()[0]}")

input_path = Input(placeholder="Path to .pth file in Team Files")
path_field = Field(
    title="Path to weights file",
    description="Copy path in Team Files",
    content=input_path,
)

radio_tabs = RadioTabs(
    titles=["Pretrained models", "Custom weights"],
    contents=[
        Container(widgets=[arch_select, table, text]),
        path_field,
    ],
)

card = Card(
    title=f"3️⃣{task_selector.get_value()} models",
    description="Choose model architecture and how weights should be initialized",
    content=radio_tabs,
)


@task_selector.value_changed
def update_architecture():
    arch_select.set(get_architectures_by_task(task_selector.get_value()))


@arch_select.value_changed
def update_models(selected_arch):
    table.set_data(
        columns=titles,
        rows=get_models_by_architecture(titles, selected_arch),
        subtitles=subtitles,
    )
    table.select_row(0)
    text.text = f"selected model: {table.get_selected_row()[0]}"


@table.value_changed
def update_selected_model(selected_row):
    text.text = f"selected model: {selected_row[0]}"
