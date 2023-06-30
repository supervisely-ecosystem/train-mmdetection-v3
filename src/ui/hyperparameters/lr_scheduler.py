from typing import List
from supervisely.app.widgets import (
    Select,
    Container,
    Field,
    Input,
    InputNumber,
    Empty,
    SelectString,
    Switch,
)

from src.ui.utils import (
    OrderedWidgetWrapper,
    set_switch_value,
    get_switch_value,
    create_linked_getter,
)
from src.train_parameters import TrainParameters

schedulers = [("empty", "Without scheduler")]

by_epoch_input = Switch(True)
by_epoch_field = Field(by_epoch_input, "By epoch")

# Step scheduler
step_scheduler = OrderedWidgetWrapper("StepLR")
step_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

step_input = InputNumber(3, 1, step=1)
step_field = Field(step_input, "LR sheduler step")
step_scheduler.add_input(
    "step_size",
    step_input,
    wraped_widget=step_field,
)

step_gamma_input = InputNumber(0.1, 0, step=1e-5, size="small")
step_gamma_field = Field(step_gamma_input, "Gamma")
step_scheduler.add_input("gamma", step_gamma_input, step_gamma_field)
schedulers.append((repr(step_scheduler), "Step LR"))


# Multistep
def get_multisteps(input_w: Input) -> List[int]:
    steps: str = input_w.get_value()
    return [int(st.strip()) for st in steps.split(",")]


def set_multisteps(input_w: Input, value: List[int]):
    input_w.set_value(",".join(value))


multi_steps_scheduler = OrderedWidgetWrapper("MultiStepLR")
multi_steps_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

multi_steps_input = Input("16,22")
multi_steps_field = Field(
    multi_steps_input,
    "LR sheduler steps",
    "Many int step values splitted by comma",
)
multi_steps_scheduler.add_input(
    "milestones",
    multi_steps_input,
    wraped_widget=multi_steps_field,
    custom_value_getter=get_multisteps,
    custom_value_setter=set_multisteps,
)

multi_steps_gamma_input = InputNumber(0.1, 0, step=1e-5, size="small")
multi_steps_gamma_field = Field(multi_steps_gamma_input, "Gamma")
multi_steps_scheduler.add_input("gamma", multi_steps_gamma_input, multi_steps_gamma_field)
schedulers.append((repr(multi_steps_scheduler), "Multistep LR"))

# exponential
exp_scheduler = OrderedWidgetWrapper("ExponentialLR")
exp_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

exp_gamma_input = InputNumber(0.1, 0, step=1e-5, size="small")
exp_gamma_field = Field(exp_gamma_input, "Gamma")
exp_scheduler.add_input("gamma", exp_gamma_input, exp_gamma_field)
schedulers.append((repr(exp_scheduler), "Exponential LR"))


# reduce on plateau
reduce_plateau_scheduler = OrderedWidgetWrapper("ReduceOnPlateauLR")
reduce_plateau_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

reduce_plateau_factor_input = InputNumber(0.1, 0, step=1e-5, size="small")
reduce_plateau_factor_field = Field(
    reduce_plateau_factor_input,
    "Factor",
    "Factor by which the learning rate will be reduced. new_param = param * factor",
)
reduce_plateau_scheduler.add_input(
    "factor",
    reduce_plateau_factor_input,
    reduce_plateau_factor_field,
)

reduce_plateau_patience_input = InputNumber(10, 2, step=1, size="small")
reduce_plateau_patience_field = Field(
    reduce_plateau_patience_input,
    "Patience",
    "Number of epochs with no improvement after which learning rate will be reduced",
)
reduce_plateau_scheduler.add_input(
    "patience",
    reduce_plateau_patience_input,
    reduce_plateau_patience_field,
)

schedulers.append((repr(reduce_plateau_scheduler), "ReduceOnPlateau LR"))

# CosineAnnealingLR
cosineannealing_scheduler = OrderedWidgetWrapper("CosineAnnealingLR")
cosineannealing_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

# TODO: нужен ли Tmax и если да, то какое дефолтное значение
cosineannealing_tmax_input = InputNumber(1, 1, step=1, size="small")
cosineannealing_tmax_field = Field(
    cosineannealing_tmax_input,
    "T max",
    "Maximum number of iterations",
)
cosineannealing_scheduler.add_input(
    "T_max",
    cosineannealing_tmax_input,
    cosineannealing_tmax_field,
)

etamin_switch_input = Switch(True)
etamin_input = InputNumber(0, 0, step=1e-6, size="small")
etamin_field = Field(
    Container([etamin_switch_input, etamin_input]),
    "Min LR",
    "Minimum learning rate",
)

etamin_ratio_input = InputNumber(0, 0, step=1e-6, size="small")
etamin_ratio_input.disable()
etamin_ratio_field = Field(
    etamin_ratio_input,
    "Min LR Ratio",
    "The ratio of the minimum parameter value to the base parameter value",
)

cosineannealing_scheduler.add_input(
    "eta_min",
    etamin_input,
    etamin_field,
    custom_value_getter=create_linked_getter(
        etamin_input,
        etamin_ratio_input,
        etamin_switch_input,
        True,
    ),
)

cosineannealing_scheduler.add_input(
    "eta_min_ratio",
    etamin_ratio_input,
    etamin_ratio_field,
    custom_value_getter=create_linked_getter(
        etamin_input,
        etamin_ratio_input,
        etamin_switch_input,
        False,
    ),
)
schedulers.append((repr(cosineannealing_scheduler), "Cosine Annealing LR"))


# CosineRestartLR
cosinerestart_scheduler = OrderedWidgetWrapper("CosineRestartLR")
cosinerestart_scheduler.add_input(
    "by_epoch",
    by_epoch_input,
    by_epoch_field,
    get_switch_value,
    set_switch_value,
)

cosinerestart_preiods_input = Input("1")
cosinerestart_preiods_field = Field(
    cosinerestart_preiods_input,
    "Periods",
    "Periods for each cosine anneling cycle. Many int step values splitted by comma",
)
cosinerestart_scheduler.add_input(
    "periods",
    cosinerestart_preiods_input,
    wraped_widget=cosinerestart_preiods_field,
    custom_value_getter=get_multisteps,
    custom_value_setter=set_multisteps,
)

cosinerestart_restart_weights_input = Input("1")
cosinerestart_restart_weights_field = Field(
    cosinerestart_preiods_input,
    "Restart weights",
    "Periods for each cosine anneling cycle. Many int step values splitted by comma",
)

cosinerestart_scheduler.add_input(
    "restart_weights",
    cosinerestart_restart_weights_input,
    wraped_widget=cosinerestart_restart_weights_field,
    custom_value_getter=get_multisteps,
    custom_value_setter=set_multisteps,
)

cosinerestart_scheduler.add_input(
    "eta_min",
    etamin_input,
    etamin_field,
    custom_value_getter=create_linked_getter(
        etamin_input,
        etamin_ratio_input,
        etamin_switch_input,
        True,
    ),
)

cosinerestart_scheduler.add_input(
    "eta_min_ratio",
    etamin_ratio_input,
    etamin_ratio_field,
    custom_value_getter=create_linked_getter(
        etamin_input,
        etamin_ratio_input,
        etamin_switch_input,
        False,
    ),
)
schedulers.append((repr(cosinerestart_scheduler), "Cosine Restart LR"))

# LinearLR
# PolyLR
# OneCycleLR


# warmup
enable_warmup_input = Switch(True)
enable_warmup_field = Field(enable_warmup_input, "Enable warmup")

warmup_strategy = ["linear", "constant", "exp"]
warmup = OrderedWidgetWrapper("warmup")
warmup_selector = SelectString(warmup_strategy, warmup_strategy)
warmup_strategy_field = Field(warmup_selector, "Warmup strategy")
warmup.add_input(
    "warmup_strategy",
    warmup_selector,
    warmup_strategy_field,
    custom_value_getter=lambda w: w.get_value(),
    custom_value_setter=lambda w, v: w.set_value(v),
)

warmup_iterations = InputNumber(400, 0)
warmup_iterations_field = Field(
    warmup_iterations, "Warmup iterations", "The number of iterations that warmup lasts"
)
warmup.add_input("warmup_steps", warmup_iterations, warmup_iterations_field)

warmup_ratio = InputNumber(0.001, step=1e-4)
warmup_ratio_field = Field(
    warmup_ratio,
    "Warmup ratio",
    "LR used at the beginning of warmup equals to warmup_ratio * initial_lr",
)
warmup.add_input("warmup_ratio", warmup_ratio, warmup_ratio_field)


# Total
schedulers_params = {
    "Without scheduler": Empty(),
    repr(step_scheduler): step_scheduler,
    repr(multi_steps_scheduler): multi_steps_scheduler,
    repr(exp_scheduler): exp_scheduler,
    repr(cosineannealing_scheduler): cosineannealing_scheduler,
    repr(reduce_plateau_scheduler): reduce_plateau_scheduler,
    repr(cosinerestart_scheduler): cosinerestart_scheduler,
}

select_scheduler = Select([Select.Item(val, label) for val, label in schedulers])
select_scheduler_field = Field(
    select_scheduler,
    title="Select scheduler",
)

schedulres_tab = Container(
    [
        select_scheduler_field,
        step_scheduler.create_container(hide=True),
        multi_steps_scheduler.create_container(hide=True),
        exp_scheduler.create_container(hide=True),
        reduce_plateau_scheduler.create_container(hide=True),
        cosineannealing_scheduler.create_container(hide=True),
        cosinerestart_scheduler.create_container(hide=True),
        enable_warmup_field,
        warmup.create_container(),
    ]
)


def get_scheduler_params() -> OrderedWidgetWrapper:
    name = select_scheduler.get_value()
    return schedulers_params[name]


def update_scheduler_widgets_with_params(params: TrainParameters):
    # scheduler
    if params.scheduler is None:
        select_scheduler.set_value("empty")
    else:
        name = params.scheduler["type"]
        select_scheduler.set_value(name)

        for param, value in select_scheduler.items():
            if param in schedulers_params[name].get_params():
                schedulers_params[name].set(param, value)

    # warmup
    if params.warmup_steps:
        enable_warmup_input.on()
    else:
        enable_warmup_input.off()
    warmup.set("warmup_strategy", params.warmup_strategy)
    warmup.set("warmup_steps", params.warmup_steps)
    warmup.set("warmup_ratio", params.warmup_ratio)


def update_scheduler_params_with_widgets(params: TrainParameters) -> TrainParameters:
    params.warmup_strategy = warmup.warmup_strategy
    if enable_warmup_input.is_switched():
        params.warmup_steps = warmup.warmup_steps
    else:
        params.warmup_steps = 0
    params.warmup_ratio = warmup.warmup_ratio

    name = select_scheduler.get_value()
    if name == "empty":
        params.scheduler = None
        return params

    scheduler_widget: OrderedWidgetWrapper = schedulers_params[name]
    new_params = scheduler_widget.get_params()
    new_params["type"] = repr(scheduler_widget)
    params.scheduler = new_params

    return params
