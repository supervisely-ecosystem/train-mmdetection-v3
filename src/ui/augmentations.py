from supervisely.app.widgets import AugmentationsWithTabs, Card, Container

from src.ui.task import task_selector


_templates = [{"value": "import всякое from конкретного", "label": "Light"}]
if "segmentation" in task_selector.get_value().lower():
    task_type = "segmentation"
else:
    task_type = "detection"

augments = AugmentationsWithTabs(task_type=task_type, templates=_templates)
card = Card(
    title="5️⃣Training augmentations",
    description="Choose one of the prepared templates or provide custom pipeline",
    content=augments,
)
