from typing import List
from supervisely.app.widgets import Select, Container, Field, Input, InputNumber, Empty

from src.ui.utils import OrderedWidgetWrapper
from src.train_parameters import TrainParameters

schedulers = [("empty", "Without scheduler")]


# Step scheduler
step_scheduler = OrderedWidgetWrapper("StepParamScheduler")
step_input = InputNumber(3, 1, step=1)
step_field = Field(step_input, "LR sheduler step")
step_scheduler.add_input(
    "step_size",
    step_input,
    wraped_widget=step_field,
)

step_gamma_input = InputNumber(0.1, 0, step=1e-5, size="medium")
step_gamma_field = Field(step_gamma_input, "Gamma")
step_scheduler.add_input("gamma", step_gamma_input, step_gamma_field)
schedulers.append((repr(step_scheduler), "Step LR"))


# Multistep
def get_multisteps(input_w: Input) -> List[int]:
    steps: str = input_w.get_value()
    return [int(st.strip()) for st in steps.split(",")]


def set_multisteps(input_w: Input, value: List[int]):
    input_w.set_value(",".join(value))


multi_steps_scheduler = OrderedWidgetWrapper("MultiStepParamScheduler")
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

multi_steps_gamma_input = InputNumber(0.1, 0, step=1e-5, size="medium")
multi_steps_gamma_field = Field(multi_steps_gamma_input, "Gamma")
multi_steps_scheduler.add_input("gamma", multi_steps_gamma_input, multi_steps_gamma_field)
schedulers.append((repr(multi_steps_scheduler), "Multistep LR"))

# exponential
exp_scheduler = OrderedWidgetWrapper("ExponentialParamScheduler")
exp_gamma_input = InputNumber(0.1, 0, step=1e-5, size="medium")
exp_gamma_field = Field(exp_gamma_input, "Gamma")
exp_scheduler.add_input("gamma", exp_gamma_input, exp_gamma_field)
schedulers.append((repr(exp_scheduler), "Exponential LR"))


# warmup

warmup_strategy = ["constant", "linear", "exp"]
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

warmup_iterations = InputNumber(0, 0)
warmup_iterations_field = Field(
    warmup_iterations, "Warmup iterations", "The number of iterations that warmup lasts"
)
warmup.add_input("warmup_steps", warmup_iterations, warmup_iterations_field)

warmup_ratio = InputNumber(0, 0)
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
}

select_scheduler = Select([Select.Item(val, label) for val, label in schedulers])

schedulres_tab = Container(
    [
        select_scheduler,
        step_scheduler.create_container(hide=True),
        multi_steps_scheduler.create_container(hide=True),
        exp_scheduler.create_container(hide=True),
        warmup,
    ]
)


def get_scheduler_params() -> OrderedWidgetWrapper:
    name = select_scheduler.get_value()
    return schedulers_params[name]


def update_scheduler_widgets_with_params(params: TrainParameters):
    if params.scheduler is None:
        select_scheduler.set_value("empty")

    name = params.scheduler["type"]
    select_scheduler.set_value(name)

    for param, value in params.select_scheduler.items():
        if param in schedulers_params[name].get_params():
            schedulers_params[name].set(param, value)

    warmup.set("warmup_strategy", params.warmup_strategy)
    warmup.set("warmup_steps", params.warmup_steps)
    warmup.set("warmup_ratio", params.warmup_ratio)


def update_scheduler_params_with_widgets(params: TrainParameters) -> TrainParameters:
    params.warmup_strategy = warmup.warmup_strategy
    params.warmup_steps = warmup.warmup_steps
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
