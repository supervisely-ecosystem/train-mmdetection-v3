from supervisely.app.widgets import Select, InputNumber, Switch, Field, Container

from src.ui.utils import InputContainer


def switch_get_value(switch: Switch):
    return switch.is_switched()


optimizers_params = InputContainer()
optimizers_names = ["Adam", "AdamW", "SGD"]

selector_input = Select([Select.Item(name, name) for name in optimizers_names])
selector_field = Field(
    selector_input,
    title="Optimizer",
    description=(
        "Choose optimizer and its settings, learn " "more in official pytorch documentation"
    ),
)
# TODO: ссылку на доки в дескрипшн
# '<a href="https://pytorch.org/docs/1.10.0/optim.html#algorithms">documentation</a>'

optimizers_params.add_input("optimizer", selector_input)

lr_input = InputNumber(0.01, 0, step=5e-5)
lr_field = Field(lr_input, title="Learning rate")
optimizers_params.add_input("lr", lr_input)

weight_decay_input = InputNumber(1e-4, 0, step=1e-5)
weight_decay_field = Field(weight_decay_input, title="Weight decay")
optimizers_params.add_input("weight_decay", weight_decay_input)

apply_clip_input = Switch(True)
clip_input = InputNumber(0.1, 0, step=1e-2)
clip_container = Container([apply_clip_input, clip_input])
clip_field = Field(
    clip_container,
    title="Clip gradient norm",
    description="Select the highest gradient norm value.",
)
optimizers_params.add_input("apply_clip", apply_clip_input, switch_get_value)
optimizers_params.add_input("max_norm", clip_input)


# Adam and AdamW
beta1_input = InputNumber(0.9, 0, step=1e-5)
beta1_field = Field(beta1_input, title="Beta 1")
optimizers_params.add_input("beta1", beta1_input)

beta2_input = InputNumber(0.999, 0, step=1e-5)
beta2_field = Field(beta2_input, title="Beta 2")
optimizers_params.add_input("beta2", beta2_input)

adams_w_fields = [beta1_field, beta2_field]

# Adam
amsgrad_input = Switch()
amsgrad_field = Field(amsgrad_input, title="Amsgrad")
optimizers_params.add_input("beta2", amsgrad_input, switch_get_value)

adam_fields = [amsgrad_field]

# SGD
momentum_input = InputNumber(0.9, 0, step=1e-2)
momentum_field = Field(momentum_input, title="Momentum")
optimizers_params.add_input("momentum", momentum_input)
momentum_field.hide()

sgd_fields = [momentum_field]

optimizers_tab = Container(
    [
        selector_field,
        lr_field,
        weight_decay_field,
        beta1_field,
        beta2_field,
        amsgrad_field,
        momentum_field,
        clip_field,
    ]
)
