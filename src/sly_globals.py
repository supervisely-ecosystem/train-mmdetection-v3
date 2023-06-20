import os
import supervisely as sly
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/supervisely.env"))

PROJECT_ID = 18337
TEAM_ID = 453

api: sly.Api = sly.Api.from_env()
