from supervisely.app.widgets import Card, Container, RadioGroup, Button, NotificationBox, Field

from src.ui.utils import update_custom_button_params

msg = """Select the task you are going to solve.
    Object detection: the model will predict bounding boxes of objects
    and all annotations will be converted to Rectangles.
    Instance Segmentation: the model will predict bounding boxes and masks of the objects.
    Only Bitmap and Polygon annotations will be used."""

info = NotificationBox(title="INFO: How to select task?", description=msg, box_type="info")
task_selector = RadioGroup(
    items=[
        RadioGroup.Item(value="Object detection", label="Object detection"),
        RadioGroup.Item(value="Instance Segmentation", label="Instance Segmentation"),
    ],
    direction="vertical",
)


select_field = Field(title="Select deep learning problem to solve", content=task_selector)
select_btn = Button(text="Select task")

card = Card(
    title="1️⃣ MMDetection task",
    description="Select task from list below",
    content=Container(widgets=[info, select_field, select_btn], direction="vertical"),
    lock_message="Please, select project and load data.",
)
