from supervisely.app.widgets import Switch, Text, Editor, Card, Container, Button, Switch, Field
from supervisely.app.content import StateJson

from src.sly_globals import PROJECT_ID

switch = Switch(False)
switch_field = Field(
    title="Advanced usage (not recommended for most of the users)",
    description="Do it at your own risk - training can crash if configs are incorrect",
    content=switch,
)

editor = Editor(
    height_lines=100,
    language_mode="python",
    readonly=True,
    auto_format=True,
)

filter_images_without_gt_input = Switch(True)
filter_images_without_gt_field = Field(
    filter_images_without_gt_input,
    title="Filter images without annotations",
    description="After selecting classes, some images may not have any annotations. Whether to remove them?",
)

select_btn = Button("Select")
card = Card(
    title="Training Config",
    description="Review and edit the training config",
    content=Container([switch_field, editor, select_btn]),
)
card.lock("Select hyperparameters to unlock.")

@switch.value_changed
def on_train_config_switch_changed(value):
    if value:
        editor.readonly = False
    else:
        editor.readonly = True
