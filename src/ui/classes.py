from supervisely.app.widgets import ClassesTable, Card, Container, Button
from supervisely.app.content import StateJson

from src.sly_globals import PROJECT_ID


def select_all(cls_tbl: ClassesTable):
    cls_tbl._global_checkbox = True
    cls_tbl._checkboxes = [True] * len(cls_tbl._table_data)
    StateJson()[cls_tbl.widget_id]["global_checkbox"] = cls_tbl._global_checkbox
    StateJson()[cls_tbl.widget_id]["checkboxes"] = cls_tbl._checkboxes
    StateJson().send_changes()


classes = ClassesTable(project_id=PROJECT_ID)
select_all(classes)
# select_btn = Button(text="Select classes")
# select_btn.disable()

card = Card(
    title="3️⃣Training classes",
    description=(
        "Select classes that will be used for training. "
        "Supported shapes are Bitmap, Polygon, Rectangle."
    ),
    content=classes,
)


# @classes.value_changed
# def confirmation_message(selected_classes):
#     selected_num = len(selected_classes)
#     if selected_num == 0:
#         select_btn.disable()
#         select_btn.text = "Select classes"
#     else:
#         select_btn.enable()
#         select_btn.text = f"Use {selected_num} selected classes"
