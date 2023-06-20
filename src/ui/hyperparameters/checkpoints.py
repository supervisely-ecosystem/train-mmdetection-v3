from supervisely.app.widgets import (
    Container,
    InputNumber,
    Field,
    Text,
    Switch,
)

from src.ui.utils import InputContainer
from src.ui.hyperparameters.base_params import NUM_EPOCHS


# checkpoints
checkpoint_params = InputContainer()

checkpoint_interval_input = InputNumber(1, 1, NUM_EPOCHS)
checkpoint_interval_text = Text(
    f"Save checkpoint every {checkpoint_interval_input.get_value()} epochs",
    status="info",
)
checkpoint_interval = Container([checkpoint_interval_input, checkpoint_interval_text])
checkpoint_interval_field = Field(checkpoint_interval, title="Checkpoints interval")
checkpoint_params.add_input("interval", checkpoint_interval_input)


def switch_get_value(switch: Switch):
    return switch.is_switched()


checkpoint_save_count_input = InputNumber(1, 1)
checkpoint_save_switch = Switch(switched=True)
checkpoint_save_count = Container([checkpoint_save_switch, checkpoint_save_count_input])
checkpoint_save_count_field = Field(
    checkpoint_save_count,
    title="Checkpoints save count",
    description=(
        "The maximum checkpoints to keep. "
        "In some cases we want only the latest "
        "few checkpoints and would like to delete "
        "old ones to save the disk space. "
        "If option is disabled then it means unlimited."
    ),
)
checkpoint_params.add_input("max_saves", checkpoint_save_count_input)
checkpoint_params.add_input("saves_limited", checkpoint_save_switch, switch_get_value)

checkpoint_last_switch = Switch()
checkpoint_last_field = Field(
    checkpoint_last_switch,
    title="Save last checkpoint",
    description="Whether to force the last checkpoint to be saved regardless of interval",
)
checkpoint_params.add_input("save_last", checkpoint_last_switch, switch_get_value)

checkpoint_best_switch = Switch(True)
checkpoint_best_field = Field(
    checkpoint_best_switch,
    title="Save last checkpoint",
    description="Whether to force the last checkpoint to be saved regardless of interval",
)
checkpoint_params.add_input("save_best", checkpoint_best_switch, switch_get_value)


checkpoints_tab = Container(
    [
        checkpoint_interval_field,
        checkpoint_save_count_field,
        checkpoint_last_field,
        checkpoint_best_field,
    ]
)
