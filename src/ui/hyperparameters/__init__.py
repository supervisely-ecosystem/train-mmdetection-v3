from supervisely.app.widgets import Tabs, Card

import src.ui.hyperparameters.handlers as handlers
from src.ui.hyperparameters.checkpoints import checkpoints_tab, checkpoint_params
from src.ui.hyperparameters.general import general_tab, general_params
from src.ui.hyperparameters.optimizers import optimizers_tab, optimizers_params


content = Tabs(
    labels=["General", "Checkpoints", "Optimizers"],
    contents=[general_tab, checkpoints_tab, optimizers_tab],
)

card = Card(
    title="Training hyperparameters",
    description="Partially taken from default model configs",
    content=content,
)
