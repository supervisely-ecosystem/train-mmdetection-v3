from typing import Optional
from supervisely.app.widgets import Select, SelectString, InputNumber, Switch, Field, Container

from src.ui.utils import InputContainer, OrderedWidgetWrapper


def switch_get_value(switch: Switch):
    return switch.is_switched()


optimizers_names = []

lr = InputNumber(0.01, 0, step=5e-5)
lr_field = Field(lr, title="Learning rate")

wd = InputNumber(1e-4, 0, step=1e-5)
wd_field = Field(wd, title="Weight decay")

adam_beta1 = InputNumber(0.9, 0, step=1e-5)
adam_beta1_field = Field(adam_beta1, title="Beta 1")

adam_beta2 = InputNumber(0.999, 0, step=1e-5)
adam_beta2_field = Field(adam_beta2, title="Beta 2")

amsgrad_input = Switch()
amsgrad_field = Field(amsgrad_input, title="Amsgrad")

sgd_momentum = InputNumber(0.9, 0, step=1e-2)
sgd_momentum_field = Field(sgd_momentum, title="Momentum")


adam = OrderedWidgetWrapper("Adam")
adam.add_input("lr", lr, lr_field)
adam.add_input("weight_decay", wd, wd_field)
adam.add_input("beta1", adam_beta1, adam_beta1_field)
adam.add_input("beta2", adam_beta2, adam_beta2_field)
adam.add_input("amsgrad", amsgrad_input, amsgrad_field, switch_get_value)
optimizers_names.append(repr(adam))


adamw = OrderedWidgetWrapper("AdamW")
adamw.add_input("lr", lr, lr_field)
adamw.add_input("weight_decay", wd, wd_field)
adamw.add_input("beta1", adam_beta1, adam_beta1_field)
adamw.add_input("beta2", adam_beta2, adam_beta2_field)
optimizers_names.append(repr(adamw))


sgd = OrderedWidgetWrapper("SGD")
sgd.add_input("lr", lr, lr_field)
sgd.add_input("weight_decay", wd, wd_field)
sgd.add_input("momentum", sgd_momentum, sgd_momentum_field)
optimizers_names.append(repr(sgd))

optimizers_params = {
    repr(adam): adam,
    repr(adamw): adamw,
    repr(sgd): sgd,
}

link = "https://pytorch.org/docs/1.10.0/optim.html#algorithms"
select_optim = SelectString(
    optimizers_names,
    optimizers_names,
    items_links=[link for _ in range(len(optimizers_names))],
)


apply_clip_input = Switch(True)
clip_input = InputNumber(0.1, 0, step=1e-2)
clip_container = Container([apply_clip_input, clip_input])
clip_field = Field(
    clip_container,
    title="Clip gradient norm",
    description="Select the highest gradient norm value.",
)

optimizers_tab = Container(
    [
        select_optim,
        adam.create_container(),
        adamw.create_container(True),
        sgd.create_container(True),
        clip_field,
    ]
)


def get_optimizer_params() -> OrderedWidgetWrapper:
    name = select_optim.get_value()
    return optimizers_params[name]


def get_clip() -> Optional[float]:
    if apply_clip_input.is_switched():
        return clip_input.get_value()
    return None
