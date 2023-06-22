from typing import List
from supervisely.app.widgets import Widget

from src.ui.hyperparameters.checkpoints import checkpoint_interval_text, checkpoint_interval_input
from src.ui.hyperparameters.general import (
    logfreq_text,
    val_text,
    logfreq_input,
    validation_input,
    epochs_input,
)

from src.ui.hyperparameters.optimizers import (
    selector_input,
    adams_w_fields,
    adam_fields,
    sgd_fields,
    apply_clip_input,
    clip_input,
)

from src.ui.hyperparameters.lr_scheduler import (
    schedulers_params,
    select_scheduler,
)


def show_hide_fields(fields: List[Widget], hide: bool = True):
    for field in fields:
        if hide:
            field.hide()
        else:
            field.show()


@epochs_input.value_changed
def epoch_num_changes(new_value):
    validation_input.max = new_value
    checkpoint_interval_input.max = new_value


@validation_input.value_changed
def update_validation_desc(new_value):
    val_text.text = f"Evaluate validation set every {new_value} epochs"


@logfreq_input.value_changed
def update_logging_desc(new_value):
    logfreq_text.text = f"Log metrics every {new_value} iterations"


@checkpoint_interval_input.value_changed
def update_ch_interval_desc(new_value):
    checkpoint_interval_text.text = f"Save checkpoint every {new_value} epochs"


@apply_clip_input.value_changed
def enable_disable_clip(new_value):
    if new_value:
        clip_input.enable()
    else:
        clip_input.disable()


@selector_input.value_changed
def hide_optim_parameters(new_optim):
    if "adamw" in new_optim.lower():
        show_hide_fields(sgd_fields)
        show_hide_fields(adam_fields)
        show_hide_fields(adams_w_fields, False)
    elif "adam" in new_optim.lower():
        show_hide_fields(sgd_fields)
        show_hide_fields(adams_w_fields, False)
        show_hide_fields(adam_fields, False)
    else:
        show_hide_fields(adam_fields)
        show_hide_fields(adams_w_fields)
        show_hide_fields(sgd_fields, False)


@select_scheduler.value_changed
def update_scheduler(new_value):
    for scheduler in schedulers_params.keys():
        if new_value == scheduler:
            schedulers_params[scheduler].show()
        else:
            schedulers_params[scheduler].hide()
