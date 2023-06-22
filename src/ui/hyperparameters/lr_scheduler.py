from typing import List
from supervisely.app.widgets import Select, Container, Field, Input, InputNumber

from src.ui.utils import OrderedWidgetWrapper

schedulers = []


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

# step_min_lr_input = InputNumber(0, 0, step=1e-6, size="medium")
# step_min_lr_field = Field(step_min_lr_input, "Min LR")
# step_scheduler.add_input("min_lr", step_min_lr_input, step_min_lr_field)
schedulers.append((repr(step_scheduler), "Step LR"))


# Multistep
def get_multisteps(input_w: Input) -> List[int]:
    steps: str = input_w.get_value()
    return [int(st.strip()) for st in steps.split(",")]


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
)

multi_steps_gamma_input = InputNumber(0.1, 0, step=1e-5, size="medium")
multi_steps_gamma_field = Field(multi_steps_gamma_input, "Gamma")
multi_steps_scheduler.add_input("gamma", multi_steps_gamma_input, multi_steps_gamma_field)

# step_min_lr_input = InputNumber(0, 0, step=1e-6, size="medium")
# step_min_lr_field = Field(step_min_lr_input, "Min LR")
# step_scheduler.add_input("min_lr", step_min_lr_input, step_min_lr_field)
schedulers.append((repr(multi_steps_scheduler), "Multistep LR"))

# exponential
exp_scheduler = OrderedWidgetWrapper("ExponentialParamScheduler")
exp_gamma_input = InputNumber(0.1, 0, step=1e-5, size="medium")
exp_gamma_field = Field(exp_gamma_input, "Gamma")
exp_scheduler.add_input("gamma", exp_gamma_input, exp_gamma_field)
schedulers.append((repr(exp_scheduler), "Exponential LR"))


schedulers_params = {
    repr(step_scheduler): step_scheduler,
    repr(multi_steps_scheduler): multi_steps_scheduler,
    repr(exp_scheduler): exp_scheduler,
}

select_scheduler = Select([Select.Item(val, label) for val, label in schedulers])

schedulres_tab = Container(
    [
        select_scheduler,
        step_scheduler.create_container(),
        multi_steps_scheduler.create_container(hide=True),
        exp_scheduler.create_container(hide=True),
    ]
)


def get_scheduler_params() -> OrderedWidgetWrapper:
    name = select_scheduler.get_value()
    return schedulers_params[name]
