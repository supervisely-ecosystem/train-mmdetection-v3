from typing import Tuple
from supervisely.app.widgets import (
    Container,
    InputNumber,
    BindedInputNumber,
    Select,
    Field,
    Text,
)

from src.ui.utils import InputContainer
from src.ui.hyperparameters.base_params import NUM_EPOCHS

# General
general_params = InputContainer()
device_input = Select([Select.Item(0, 0), Select.Item(1, 1)])
device_field = Field(
    device_input,
    title="Device",
    description=(
        "Run nvidia-smi or check agent page to "
        "see how many devices your machine has "
        "or keep by default"
    ),
)
general_params.add_input("device", device_input)

epochs_input = InputNumber(NUM_EPOCHS, min=1)
epochs_field = Field(epochs_input, "Number of epochs")
general_params.add_input("epochs", epochs_input)


def size_and_prop(inp: BindedInputNumber) -> Tuple[Tuple[int, int], bool]:
    return inp.get_value(), inp.proportional


size_input = BindedInputNumber(640, 320, proportional=False)
general_params.add_input("size_and_prop", size_input, size_and_prop)

size_field = Field(
    size_input,
    title="Input size",
    description="Model input resolution",
)

validation_input = InputNumber(1, 1, general_params.epochs)
val_text = Text(
    f"Evaluate validation set every {validation_input.get_value()} epochs",
    status="info",
)
validation_interval = Container([validation_input, val_text])

validation_field = Field(
    validation_interval,
    title="Validation interval",
    description=(
        "By default we evaluate the model on the "
        "validation set after each epoch, you can "
        "change the evaluation interval"
    ),
)
general_params.add_input("validation_interval", validation_input)


logfreq_input = InputNumber(1000, 1)
logfreq_text = Text(
    f"Log metrics every {logfreq_input.get_value()} iterations",
    status="info",
)
logfreq_interval = Container([logfreq_input, logfreq_text])

logfreq_field = Field(
    logfreq_interval,
    title="Logging frequency",
    description=(
        "How often metrics should be logged, increase if training data is small (by iterations)."
    ),
)
general_params.add_input("logging_frequency", logfreq_field)

general_tab = Container([device_field, epochs_field, size_field, validation_field, logfreq_field])
