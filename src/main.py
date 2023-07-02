import src.sly_globals as g
import supervisely as sly
from supervisely.app.widgets import Container, Button

import src.ui.input_project as input_project
import src.ui.task as task
import src.ui.models as models
import src.ui.classes as classes
import src.ui.train_val_split as train_val_split
import src.ui.graphics as graphics
import src.ui.hyperparameters as hyperparameters
import src.ui.handlers
import src.ui.train as train
import src.ui.augmentations as augmentations
import src.ui.model_leaderboard as model_leaderboard


widgets = [
    input_project.card,
    task.card,
    model_leaderboard.card,
    models.card,
    classes.card,
    train_val_split.card,
    augmentations.card,
    hyperparameters.card,
    train.card,
]
layout = Container(widgets=widgets)
app = sly.Application(layout=layout)

g.app = app

# 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ 🔟
