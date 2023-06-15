from time import sleep  # remove
from supervisely.app.widgets import (
    SelectProject,
    Card,
    Container,
    Button,
    DoneLabel,
    Progress,
    Text,
    ProjectThumbnail,
)

import src.ui.task as task
from src.ui.classes import classes
from src.sly_globals import api

project_selector = SelectProject()
info = DoneLabel(text="Project has been successfully downloaded")
info.hide()

warning = Text(text="Select project before load", status="error")
warning.hide()

progress = Progress("loading")
progress.hide()

submit = Button(text="Download", show_loading=True)

project_th = ProjectThumbnail()
project_th.hide()

card = Card(
    title="1️⃣Input project",
    description="Download images and annotations from server to local app directory",
    content=Container(
        widgets=[project_selector, info, warning, progress, project_th, submit],
        direction="vertical",
    ),
    lock_message="Project selected",
)


@submit.click
def load_project_data():
    if project_selector.get_selected_id() is None:
        warning.show()
    else:
        warning.hide()

        pid = project_selector.get_selected_id()
        project_info = api.project.get_info_by_id(pid)
        classes.read_project_from_id(pid)
        project_th.set(project_info)
        project_th.show()
        # load data
        progress.show()
        for _ in progress(range(2)):
            sleep(1)

        info.show()
        card.lock()

        # TODO: locker/unlocker class for all cards
        task.card.unlock()
