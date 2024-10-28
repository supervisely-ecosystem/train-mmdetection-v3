import yaml
from fastapi import Request

import src.sly_globals as g
import src.ui.handlers as handlers
import supervisely as sly
from src.auto_train import start_auto_train
from supervisely.app.widgets import Container

layout = Container(widgets=[handlers.stepper])
app = sly.Application(layout=layout)

g.app = app


# @g.app.server.post("/auto_train")
# def auto_train(request: Request):
def auto_train(state):
    sly.logger.info("Starting automatic training session...")
    # state = request.state.state
    start_auto_train(state)


with open("/root/projects/train-mmdetection-v3/src/auto_train_params.yaml", "r") as file:
    state = yaml.safe_load(file)
auto_train(state)
