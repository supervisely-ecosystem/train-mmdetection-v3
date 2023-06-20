from random import randint
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from collections import OrderedDict
from supervisely.app.widgets import GridPlot, Button, Card, Container, Field, Slider, LinePlot
from supervisely.app.content import StateJson, DataJson

NumT = Union[int, float]


class StageMonitoring(object):
    def __init__(self, stage_id: str, title: str, description: str) -> None:
        self._name = stage_id
        self._metrics = OrderedDict()
        self._title = title
        self._description = description

    def create_metric(self, metric: str, series: Optional[List[str]] = None):
        if metric in self._metrics:
            raise ArithmeticError("Metric already exists.")

        if series is None:
            srs = []
        else:
            srs = [
                {
                    "name": ser,
                    "data": [],
                }
                for ser in series
            ]

        self._metrics[metric] = {
            "title": metric,
            "series": srs,
        }

    def create_series(self, metric: str, series: Union[List[str], str]):
        if isinstance(series, str):
            series = [series]
        new_series = [{"name": ser, "data": []} for ser in series]
        self._metrics[metric]["series"].extend(new_series)

    def compile_grid_field(self) -> Tuple[Field, GridPlot]:
        data = list(self._metrics.values())
        grid = GridPlot(data, columns=len(data))
        field = Field(grid, self._title, self._description)
        return field, grid

    @property
    def name(self):
        return self._name


class Monitoring(object):
    def __init__(self) -> None:
        self._stages = {}

    def add_stage(self, stage: StageMonitoring):
        field, grid = stage.compile_grid_field()
        self._stages[stage.name] = {}
        self._stages[stage.name]["compiled"] = field
        self._stages[stage.name]["raw"] = grid

    def add_scalar(
        self,
        stage_id: str,
        metric_name: str,
        serise_name: str,
        x: NumT,
        y: NumT,
    ):
        self._stages[stage_id]["raw"].add_scalar(f"{metric_name}/{serise_name}", y, x)

    def add_scalars(
        self,
        stage_id: str,
        metric_name: str,
        new_values: Dict[str, NumT],
        x: NumT,
    ):
        self._stages[stage_id]["raw"].add_scalars(
            metric_name,
            new_values,
            x,
        )

    def compile_monitoring_container(self) -> Container:
        container = Container([stage["compiled"] for stage in self._stages.values()])
        return container


smooth_slider = Slider(0, 0, 1, step=0.1, show_input=True, show_input_controls=True)

train_stage = StageMonitoring("train", "Train", "TRAIN")
train_stage.create_metric("Loss")
train_stage.create_series("Loss", "my_loss1")
train_stage.create_series("Loss", "my_loss2")

train_stage.create_metric("Ssol")
train_stage.create_series("Ssol", "my_ssol1")

monitoring = Monitoring()
monitoring.add_stage(train_stage)

add_btn = Button("add")

size1 = 10
x1 = list(range(size1))
y1 = np.random.randint(low=10, high=148, size=size1).tolist()
s1 = [[x, y] for x, y in zip(x1, y1)]

lp = LinePlot("mlp", [{"name": "frism", "data": s1}])  # , smoothing_weight=0.1)

size2 = 30
x2 = list(range(size2))
y2 = np.random.randint(low=0, high=300, size=size2).tolist()
s2 = [{"x": x, "y": y} for x, y in zip(x2, y2)]


card = Card(
    "6️⃣Training progress",
    "Task progress, detailed logs, metrics charts, and other visualizations",
    content=Container([monitoring.compile_monitoring_container(), add_btn, smooth_slider, lp]),
)

x = size1


@add_btn.click
def add_rdata():
    global x
    y0, y1 = randint(-3, 3), randint(-4, 4)
    monitoring.add_scalars("train", "Loss", {"my_loss1": y0, "my_loss2": y1}, x)
    y3 = randint(-5, 5)
    monitoring.add_scalar("train", "Ssol", "my_ssol1", x, y3)
    y3 = randint(10, 200)
    lp.add_to_series("frism", [x, y3])
    x += 1


@smooth_slider.value_changed
def change_smoothness(new_val):
    lp.change_smoothness(new_val)
    # lp._options["smoothingWeight"] = new_val  # int(new_val * 100)
    # DataJson()[lp.widget_id]["options"] = lp._options
    # DataJson().send_changes()
