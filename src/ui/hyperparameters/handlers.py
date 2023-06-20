from src.ui.hyperparameters.checkpoints import checkpoint_interval_text, checkpoint_interval_input
from src.ui.hyperparameters.general import (
    logfreq_text,
    val_text,
    logfreq_input,
    validation_input,
    epochs_input,
)


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
