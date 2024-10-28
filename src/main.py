from fastapi import Request

import src.sly_globals as g
import src.ui.handlers as handlers
import supervisely as sly
from src.auto_train import start_auto_train
from supervisely.app.widgets import Container

layout = Container(widgets=[handlers.stepper])
app = sly.Application(layout=layout)

g.app = app


@g.app.server.post("/auto_train")
def auto_train(request: Request):
    sly.logger.info("Starting automatic training session...")
    state = request.state.state
    start_auto_train(state)
