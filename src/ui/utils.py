from typing import Dict, Any
from supervisely.app import DataJson
from supervisely.app.widgets import Button

import src.ui.task as task


def update_custom_button_params(
    btn: Button,
    params_dct: Dict[str, Any],
) -> None:
    btn_state = btn.get_json_data()
    for key in params_dct.keys():
        if key not in btn_state:
            raise AttributeError(f"Parameter {key} doesn't exists.")
        if key == "icon":
            new_icon = f'<i class="{params_dct[key]}" style="margin-right: {btn._icon_gap}px"></i>'
            DataJson()[btn.widget_id][key] = new_icon
        else:
            DataJson()[btn.widget_id][key] = params_dct[key]
    DataJson().send_changes()
