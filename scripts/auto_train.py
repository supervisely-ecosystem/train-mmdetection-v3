import os
from pathlib import Path
from time import sleep

import requests
from dotenv import load_dotenv

import supervisely as sly

load_dotenv("supervisely.env")
load_dotenv("local.env")

api = sly.Api()

# get env variables
GLOBAL_TIMEOUT = 1  # seconds
AGENT_ID = 359  # agent id to run training on
APP_NAME = "supervisely-ecosystem/train-mmdetection-v3"
PROJECT_ID = sly.env.project_id()
TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
DATA_DIR = sly.app.get_data_dir()
task_type = "object detection"  # or "instance segmentation"

BRANCH = True
APP_VERSION = "auto-train-impl-v2"

##################### PART 1: TRAINING #####################

module_id = api.app.get_ecosystem_module_id(APP_NAME)
module_info = api.app.get_ecosystem_module_info(module_id)
project_name = api.project.get_info_by_id(PROJECT_ID).name

sly.logger.info(f"Starting AutoTrain for application {module_info.name}")

params = module_info.get_arguments(images_project=PROJECT_ID)

session = api.app.start(
    agent_id=AGENT_ID,
    module_id=module_id,
    workspace_id=WORKSPACE_ID,
    description=f"AutoTrain session for {module_info.name}",
    task_name="AutoTrain/train",
    params=params,
    app_version=APP_VERSION,
    is_branch=BRANCH,
)

task_id = session.task_id
domain = sly.env.server_address()
token = api.task.get_info_by_id(task_id)["meta"]["sessionToken"]
post_shutdown = f"{domain}/net/{token}/sly/shutdown"

is_ready = api.app.is_ready_for_api_calls(task_id)
if not is_ready:
    api.app.wait_until_ready_for_api_calls(task_id)
sleep(10)  # still need a time after status changed

sly.logger.info(f"Session started: #{task_id}")


train_parameters = {
    "input": {"project_id": 42201, "use_cache": True},
    "model": {
        "task_type": "Object detection",
        "arch_type": "DINO",
        "model_source": "Pretrained models",
        "model_name": "dino-5scale_swin-l_8xb2-12e_coco.py",
        "train_mode": "finetune",
    },
    "hyperparameters": {
        "general": {
            "n_epochs": 20,
            "input_image_size": [1000, 600],
            "train_batch_size": 2,
            "val_batch_size": 1,
            "val_interval": 1,
            "chart_interval": 1,
        },
        "checkpoint": {
            "checkpoint_interval": 1,
            "keep_checkpoints": True,
            "max_keep_checkpoints": 3,
            "save_last_checkpoint": True,
            "save_best_checkpoint": True,
            "save_optimizer_state": False,
        },
        "optimizer": {
            "override_frozen_stages": False,
            "type": "AdamW",
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "use_clip_grad_norm": True,
            "clip_grad_norm": 0.1,
        },
        "lr_scheduler": {
            "scheduler": "empty",
            "use_warmup": True,
            "warmup_iters": 1,
            "warmup_ratio": 0.001,
            "by_epoch": True,
        },
        "evaluation": {"model_evaluation_bm": True},
    },
}


# :green_book: You can set any parameters you want to customize training in the data field
api.task.send_request(
    task_id,
    "auto_train",
    data=train_parameters,
    timeout=10e6,
)

mmdet3_artifacts = sly.nn.artifacts.MMDetection3(TEAM_ID)
framework_folder = mmdet3_artifacts.framework_folder
model_name = train_parameters["model"]["model_name"]
team_files_folder = Path(framework_folder) / f"{task_id}_{model_name}"
weights = mmdet3_artifacts.get_weights_path(str(team_files_folder))


# if save_best_checkpoint is enabled
best = None
while best is None:
    sleep(GLOBAL_TIMEOUT)
    if api.file.dir_exists(TEAM_ID, str(weights)):
        for filename in api.file.listdir(TEAM_ID, str(weights)):
            if filename.endswith(".pth"):
                if filename.startswith("best_"):
                    best = str(weights / filename)
                    sly.logger.info(f"Checkpoint founded : {best}")

requests.post(post_shutdown)

sly.logger.info("Training completed")
sly.logger.info(
    f"The weights of trained model and other artifacts uploaded in Team Files: {str(team_files_folder)}"
)
