import supervisely as sly
from supervisely.app.widgets import TrainValSplits, Card, Container
from src.sly_globals import PROJECT_ID

splits = TrainValSplits(project_id=PROJECT_ID)

card = Card(
    title="4️⃣Train / Validation splits",
    description="Define how to split your data to train/val subsets.",
    content=splits,
)


def dump_train_val_splits():
    train_split, val_split = splits.get_splits()
    app_dir = sly.app.get_data_dir()
    sly.json.dump_json_file(train_split, f"{app_dir}/train_split.json")
    sly.json.dump_json_file(val_split, f"{app_dir}/val_split.json")
