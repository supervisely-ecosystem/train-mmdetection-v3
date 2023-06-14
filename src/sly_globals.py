import os
import supervisely as sly
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/supervisely.env"))

api: sly.Api = sly.Api.from_env()
