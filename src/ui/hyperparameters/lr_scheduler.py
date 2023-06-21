from supervisely.app.widgets import Select, Container, Field, Input, InputNumber

from src.ui.utils import OrderedWidgetWrapper

schedulers = []

step_scheduler = OrderedWidgetWrapper("step_lr")
steps_input = Input("16,22")
steps_field = Field(
    steps_input,
    "LR sheduler steps",
    "One or many int step values splitted by comma",
)
step_scheduler.add_input("steps", steps_input, wraped_widget=steps_field)

step_gamma_input = InputNumber(0.1, 0, step=1e-5, size="medium")
step_gamma_field = Field(step_gamma_input, "Gamma")
step_scheduler.add_input("gamma", step_gamma_input, step_gamma_field)

step_min_lr_input = InputNumber(0, 0, step=1e-6, size="medium")
step_min_lr_field = Field(step_min_lr_input, "Min LR")
step_scheduler.add_input("min_lr", step_min_lr_input, step_min_lr_field)
schedulers.append((repr(step_scheduler), "Step LR"))

# exponential
exp_scheduler = OrderedWidgetWrapper("exp_lr")
exp_gamma_input = InputNumber(0.1, 0, step=1e-5, size="medium")
exp_gamma_field = Field(exp_gamma_input, "Gamma")
exp_scheduler.add_input("gamma", exp_gamma_input, exp_gamma_field)
schedulers.append((repr(exp_scheduler), "Exponential LR"))


schedulers_params = {
    repr(step_scheduler): step_scheduler,
    repr(exp_scheduler): exp_scheduler,
}

select_scheduler = Select([Select.Item(val, label) for val, label in schedulers])

schedulres_tab = Container(
    [
        select_scheduler,
        step_scheduler.create_container(),
        exp_scheduler.create_container(hide=True),
    ]
)


def get_scheduler_params() -> OrderedWidgetWrapper:
    name = select_scheduler.get_value()
    return schedulers_params[name]
