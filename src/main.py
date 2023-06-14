import src.sly_globals
import supervisely as sly
from supervisely.app.widgets import Container

import src.ui.input_project as input_project
import src.ui.task as task
import src.ui.models as models

widgets = [input_project.card, task.card, models.card]
layout = Container(widgets=widgets)
app = sly.Application(layout=layout)


# 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ 🔟
