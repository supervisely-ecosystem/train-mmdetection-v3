from typing import Callable, Dict, Any, Optional
from supervisely.app import DataJson
from supervisely.app.widgets import Button, Widget


def update_custom_params(
    btn: Button,
    params_dct: Dict[str, Any],
) -> None:
    btn_state = btn.get_json_data()
    for key in params_dct.keys():
        if key not in btn_state:
            raise AttributeError(f"Parameter {key} doesn't exists.")
        else:
            DataJson()[btn.widget_id][key] = params_dct[key]
    DataJson().send_changes()


def update_custom_button_params(
    btn: Button,
    params_dct: Dict[str, Any],
) -> None:
    if "icon" in params_dct:
        new_icon = f'<i class="{params_dct["icon"]}" style="margin-right: {btn._icon_gap}px"></i>'
    params_dct["icon"] = new_icon
    update_custom_params(btn, params_dct)


class InputContainer(object):
    def __init__(self) -> None:
        self._widgets = {}
        self._custom_get_value = {}

    def add_input(
        self,
        name: str,
        widget: Widget,
        custom_value_getter: Optional[Callable[[Widget], Any]] = None,
    ) -> None:
        self._widgets[name] = widget
        if custom_value_getter is not None:
            self._custom_get_value[name] = custom_value_getter

    def __getattr__(self, __name: str) -> Any:
        if __name in self._widgets:
            return self._get_value(__name)
        raise AttributeError(
            f"Widget with name {__name} does not exists, only {self._widgets.keys()}"
        )

    def _get_value(self, name: str):
        if name in self._custom_get_value:
            widget = self._widgets[name]
            return self._custom_get_value[name](widget)
        return self._widgets[name].get_value()
