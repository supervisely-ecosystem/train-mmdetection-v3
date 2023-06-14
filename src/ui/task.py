from supervisely.app.widgets import Card, Container, RadioGroup, Button, NotificationBox, Field

from src.ui.utils import update_custom_button_params

msg = (
    "MMDetection provides tool for training models to solve "
    "different deep learning problems. Currently "
    "Supervisely supports two kind of tasks: object detection "
    "and instance segmentation. Panoptic Segmentation, "
    "Contrastive Learning or Knowledge Distillation tasks "
    "from MMDetection are not supported now. At this step you "
    "should select problem that you want to solve. Of course, "
    "you should have appropriate data with markup for this task. "
    "Available labels: bitmap masks or polygons for instance "
    "segmentation (polygons will be converted to bitmaps) and "
    "any objects except points for object detection "
    "(bounding box will be calculated automatically). "
    "Outputs of object detection models - only bounding boxes "
    "with confidence. Outputs of instance segmentation models "
    "in addition contain object masks. Selected task at this "
    "step defines models list to choose. If you want to train "
    "model based on already trained custom model, choose the appropriate task."
)

info = NotificationBox(title="INFO: How to select task?", description=msg, box_type="info")
task_selector = RadioGroup(
    items=[
        RadioGroup.Item(value="Object detection", label="object_detection"),
        RadioGroup.Item(value="Instance Segmentation", label="instance_segmentation"),
    ],
    direction="vertical",
)

select_field = Field(title="Select deep learning problem to solve", content=task_selector)
select_btn = Button(text="Select task")

select_params = {"icon": None, "plain": False, "text": "Select task"}
reselect_params = {"icon": "zmdi zmdi-refresh", "plain": True, "text": "Reselect task"}

card = Card(
    title="2️⃣MMDetection task",
    description="Select task from list below",
    content=Container(widgets=[info, select_field, select_btn], direction="vertical"),
    lock_message="Please, select project and load data.",
)
card.lock()


@select_btn.click
def select_task():
    # TODO: load task config if selected
    if select_btn._click_handled:
        task_selector.disable()
        update_custom_button_params(select_btn, reselect_params)
        select_btn._click_handled = False
    else:
        task_selector.enable()
        update_custom_button_params(select_btn, select_params)
        select_btn._click_handled = True
        # TODO: restart all steps
