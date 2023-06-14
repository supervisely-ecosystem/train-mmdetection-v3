from time import sleep  # remove
from supervisely.app.widgets import SelectProject, Card, Container, Button, Text


project_selector = SelectProject()
info = Text(text="Project has been successfully downloaded", status="success")
info.hide()
submit = Button(text="Submit", show_loading=True)

card = Card(
    title="1️⃣Input project",
    description="Download images and annotations from server to local app directory",
    content=Container(widgets=[project_selector, info, submit], direction="vertical"),
    lock_message="Project selected",
)


@submit.click
def load_project_data():
    sleep(2)  # load data
    info.show()
    card.lock()
