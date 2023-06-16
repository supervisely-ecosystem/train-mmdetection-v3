from supervisely.app.widgets import TrainValSplits, Card, Container
from src.sly_globals import PROJECT_ID

splits = TrainValSplits(project_id=PROJECT_ID)

card = Card(
    title="4️⃣Train / Validation splits",
    description="Define how to split your data to train/val subsets.",
    content=splits,
)
