import os

import supervisely as sly
from dotenv import load_dotenv
from supervisely.nn.artifacts.mmdetection import MMDetection3

from src.train_parameters import TrainParameters

load_dotenv("local.env")
load_dotenv("supervisely.env")
# load_dotenv(os.path.expanduser("~/supervisely.env"))

PROJECT_ID = sly.env.project_id()
TEAM_ID = sly.env.team_id()

if sly.is_production():
    app_session_id = os.getenv("TASK_ID")
else:
    app_session_id = 34522

api: sly.Api = sly.Api.from_env()
app_dir = sly.app.get_synced_data_dir()
app: sly.Application = None


stop_training = False
config_name: str = None
params: TrainParameters = None

# for Augmentations widget:
data_dir = app_dir
team = api.team.get_info_by_id(TEAM_ID)
project_meta: sly.ProjectMeta = sly.ProjectMeta.from_json(api.project.get_meta(PROJECT_ID))
# project_fs: sly.Project = None

sly_mmdet3 = MMDetection3(TEAM_ID)

COCO_MTERIC_KEYS = ["mAP", "mAP_50", "mAP_75"]
MAX_CLASSES_TO_SHOW_CLASSWISE_METRIC = 10

project_info = api.project.get_info_by_id(PROJECT_ID)
IMAGES_COUNT = project_info.items_count
USE_CACHE = True

cfg = None
mmdet_generated_metadata = None
train_size, val_size = None, None 