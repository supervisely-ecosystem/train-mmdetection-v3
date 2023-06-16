from supervisely.app.widgets import ClassesTable, Card, Container, Button

from src.sly_globals import PROJECT_ID

classes = ClassesTable(project_id=PROJECT_ID)
select_btn = Button(text="Select classes")
select_btn.disable()

card = Card(
    title="3️⃣Training classes",
    description=(
        "Select classes, that should be used for training. "
        "Training supports only classes of shapes Polygon and Bitmap. "
        "Other classes are ignored"
    ),
    content=Container([classes, select_btn]),
)


@classes.value_changed
def confirmation_message(selected_classes):
    selected_num = len(selected_classes)
    if selected_num == 0:
        select_btn.disable()
        select_btn.text = "Select classes"
    else:
        select_btn.enable()
        select_btn.text = f"Use {selected_num} selected classes"
