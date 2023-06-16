import src.sly_globals
import supervisely as sly
from supervisely.app.widgets import Container

# import src.ui.input_project as input_project
import src.ui.task as task
import src.ui.models as models
import src.ui.classes as classes
import src.ui.train_val_split as train_val_split

# import src.ui.augmentations as augmentations


widgets = [task.card, models.card, classes.card, train_val_split.card]  # , augmentations.card]
layout = Container(widgets=widgets)
app = sly.Application(layout=layout)


# 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ 🔟
