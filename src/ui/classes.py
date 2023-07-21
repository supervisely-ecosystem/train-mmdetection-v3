from supervisely.app.widgets import ClassesTable, Card, Container, Button, Switch, Field
from supervisely.app.content import StateJson
import supervisely as sly
import traceback

from src.sly_globals import PROJECT_ID


class DebugCard(Card):
    def lock(self, message: str = None):
        sly.logger.debug(f"Card {self} was locked")
        for line in traceback.format_stack():
            sly.logger.debug(line.strip())
        return super().lock(message)

    def unlock(self):
        sly.logger.debug(f"Card {self} was unlocked")
        for line in traceback.format_stack():
            sly.logger.debug(line.strip())
        return super().unlock()


def select_all(cls_tbl: ClassesTable):
    cls_tbl._global_checkbox = True
    cls_tbl._checkboxes = [True] * len(cls_tbl._table_data)
    StateJson()[cls_tbl.widget_id]["global_checkbox"] = cls_tbl._global_checkbox
    StateJson()[cls_tbl.widget_id]["checkboxes"] = cls_tbl._checkboxes
    StateJson().send_changes()


classes = ClassesTable(project_id=PROJECT_ID)
select_all(classes)

filter_images_without_gt_input = Switch(True)
filter_images_without_gt_field = Field(
    filter_images_without_gt_input,
    title="Filter images without annotations",
    description="After selecting classes, some images may not have any annotations. Whether to remove them?",
)

card = DebugCard(
    title="3️⃣ Training classes",
    description=(
        "Select classes that will be used for training. "
        "Supported shapes are Bitmap, Polygon, Rectangle."
    ),
    content=Container([classes, filter_images_without_gt_field]),
)

card.lock()

# @classes.value_changed
# def confirmation_message(selected_classes):
#     selected_num = len(selected_classes)
#     if selected_num == 0:
#         select_btn.disable()
#         select_btn.text = "Select classes"
#     else:
#         select_btn.enable()
#         select_btn.text = f"Use {selected_num} selected classes"
