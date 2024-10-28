import io
import os
import re

from dotenv import load_dotenv

import sly_globals as g

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import random
import threading
import uuid
from functools import partial
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruamel.yaml
import supervisely as sly
import supervisely.io.env as env
import torch
import yaml
from fastapi import Request, Response
from supervisely._utils import abs_url, is_development
from supervisely.app.widgets import (  # SelectDataset,
    Button,
    Card,
    Checkbox,
    ClassesTable,
    Collapse,
    Container,
    Dialog,
    DoneLabel,
    Editor,
    Empty,
    Field,
    FileThumbnail,
    Flexbox,
    FolderThumbnail,
    GridGallery,
    GridPlot,
    ImageSlider,
    Input,
    InputNumber,
    NotificationBox,
    Progress,
    RadioGroup,
    RadioTable,
    RadioTabs,
    RandomSplitsTable,
    ReloadableArea,
    ReportThumbnail,
    SelectDatasetTree,
    SelectString,
    SlyTqdm,
    Stepper,
    Switch,
    TaskLogs,
    Text,
    Tooltip,
    TrainValSplits,
)
from supervisely.nn import TaskType
from supervisely.nn.artifacts.yolov8 import YOLOv8
from supervisely.nn.benchmark import (
    InstanceSegmentationBenchmark,
    ObjectDetectionBenchmark,
)
from supervisely.nn.inference import SessionJSON
from ultralytics.utils.metrics import ConfusionMatrix

import sly_globals as g

import src.workflow as w
from src.early_stopping.custom_yolo import YOLO as CustomYOLO
from src.metrics_watcher import Watcher
from src.project_cached import download_project
from src.serve import YOLOv8ModelMB
from src.sly_to_yolov8 import check_bbox_exist_on_images, transform
from src.utils import custom_plot, get_eval_results_dir_name, verify_train_val_sets


# Stepper
import src.ui.input_project as input_project_ui
import src.ui.task as task_ui
import src.ui.model_leaderboard as model_leaderboard_ui
import src.ui.models as models_ui
import src.ui.classes as classes_ui
import src.ui.train_val_split as train_val_split_ui
import src.ui.augmentations as augmentations_ui
import src.ui.hyperparameters as hyperparameters_ui
import src.ui.train as train_ui

import src.ui.graphics as graphics_ui
import src.ui.handlers as handlers_ui

server = g.app.server


@server.post("/auto_train")
def auto_train(request: Request):
    sly.logger.info("Starting automatic training session...")
    state = request.state.state

    if "yaml_string" in state:
        state = yaml.safe_load(state["yaml_string"])

    # Step 1. Input project
    project_id = state["project_id"]
    use_cache = state.get("use_cache", True)
    
    # local_dir = g.root_model_checkpoint_dir
    local_dir = g.data_dir
    project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(project_id))
    # ----------------------------------------------------------------------------------------------- #

    # Step 2. Select task type
    task_type = state["task_type"]
    if task_type == "object detection":
        necessary_geometries = ["rectangle"]
    elif task_type == "instance segmentation":
        necessary_geometries = ["bitmap", "polygon"]
    
    task_ui.task_selector.set_value(task_type)
    models_ui.update_architecture(task_ui.task_selector.get_value())
    # ----------------------------------------------------------------------------------------------- #
    
    # Step 3. Model Leaderboard
    model_leaderboard_ui.update_table(models_ui.models_meta, task_type)
    # ----------------------------------------------------------------------------------------------- #
    
    # Step 4. Select arch and model
    arch_type = state["arch_type"]
    model_source = state["model_source"]
    model_name = state["model_name"]
    train_mode = state["train_mode"]  
        
    models_ui.arch_select.set_value(arch_type)
    models_ui.radio_tabs.set_active_tab(model_source)
    models_ui.table.select_row(0)
    
    if train_mode == "finetune":
        models_ui.load_from.on()
    else:
        models_ui.load_from.off()
        
    # models_ui.models_table.set_data(
        # columns=models_table_columns,
        # rows=models_table_rows,
        # subtitles=models_table_subtitles,
    # )

    local_artifacts_dir =local_dir
    sly.logger.info(f"Local artifacts dir: {local_artifacts_dir}")

    if os.path.exists(local_artifacts_dir):
        sly.fs.remove_dir(local_artifacts_dir)
    # ----------------------------------------------------------------------------------------------- #

    # Step 5. Select classes
    selected_classes = [cls.name for cls in project_meta.obj_classes]
    n_classes = len(classes_ui.classes.get_selected_classes())
    if n_classes > 1:
        sly.logger.info(f"{n_classes} classes were selected successfully")
    else:
        sly.logger.info(f"{n_classes} class was selected successfully")

    # Remove classes with unnecessary shapes
    if task_type != "object detection":
        unnecessary_classes = []
        for cls in project_meta.obj_classes:
            if (
                cls.name in selected_classes
                and cls.geometry_type.geometry_name() not in necessary_geometries
            ):
                unnecessary_classes.append(cls.name)
        if len(unnecessary_classes) > 0:
            sly.Project.remove_classes(
                project_dir, classes_to_remove=unnecessary_classes, inplace=True
            )

    # Convert project to detection task if necessary
    if task_type == "object detection":
        sly.Project.to_detection_task(project_dir, inplace=True)
    # ----------------------------------------------------------------------------------------------- #
    
    # Step 6. Split the data
    try:
        train_val_split_ui.splits._project_fs = sly.Project(project_dir, sly.OpenMode.READ)
        train_set, val_set = train_val_split_ui.splits.get_splits()
    except Exception:
        if not use_cache:
            raise
        sly.logger.warning(
            "Error during data splitting. Will try to re-download project without cache",
            exc_info=True,
        )
        download_project(
            api=g.api,
            project_id=g.project_info.id,
            project_dir=project_dir,
            use_cache=False,
            progress=progress_bar_download_project,
        )
        train_val_split_ui.splits._project_fs = sly.Project(project_dir, sly.OpenMode.READ)
        train_set, val_set = train_val_split_ui.splits.get_splits()
    verify_train_val_sets(train_set, val_set)
    # ----------------------------------------------------------------------------------------------- #

    # Step 7. Augmentations
    augs = state["augmentations"]
    # ----------------------------------------------------------------------------------------------- #
    
    # Step 8. Hyperparameters
    # General
    n_epochs = state.get("n_epochs", 20)
    input_image_size = state.get("input_image_size", [1000, 600])
    train_batch_size = state.get("train_batch_size", 2)
    val_batch_size = state.get("val_batch_size", 1)
    val_interval = state.get("val_interval", 1)
    chart_interval = state.get("chart_interval", 1)
    # Checkpoints
    checkpoint_interval = state.get("checkpoint_interval", 1)
    keep_checkpoints = state.get("keep_checkpoints" ,True)
    max_keep_checkpoints = state.get("max_keep_checkpoints", 3)
    save_last_checkpoint = state.get("save_last_checkpoint", True)
    save_best_checkpoint = state.get("save_best_checkpoint", True)
    save_optimizer_state = state.get("save_optimizer_state", False)
    # Optimizer
    override_frozen_stages = state.get("override_frozen_stages", False)
    optimizer = state.get("optimizer", "AdamW")
    lr = state.get("lr", 0.0001)
    weight_decay = state.get("weight_decay", 0.0001)
    use_clip_grad_norm = state.get("use_clip_grad_norm", True)
    clip_grad_norm = state.get("clip_grad_norm", 0.1)
    # LR Scheduler
    scheduler = state.get("scheduler", "Without scheduler")
    use_warmup = state.get("use_warmup", True)
    warmup_iters = state.get("warmup_iters", 1)
    warmup_ratio = state.get("warmup_ratio", 0.001)
    # MB
    model_evaluation_bm = state.get("model_evaluation_bm", True)
    
    # Step 8. Train
    train_ui.train()

    # download model
    model_source = "Pretrained models"

    def download_monitor(monitor, g.api: sly.Api, progress: sly.Progress):
        value = monitor
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        weights_pbar.update(progress.current)

    file_info = None

    g.stop_event = threading.Event()

    if model_source == "Pretrained models":
        if "model" not in state:
            selected_index = 0
        else:
            selected_model = state["model"]
            found_index = False
            for i, element in enumerate(models_data):
                if selected_model in element.values():
                    selected_index = i
                    found_index = True
                    break
            if not found_index:
                sly.logger.info(
                    f"Unable to find requested model: {selected_model}, switching to default"
                )
                selected_index = 0
        selected_dict = models_data[selected_index]
        weights_url = selected_dict["weights_url"]
        model_filename = weights_url.split("/")[-1]
        selected_model_name = selected_dict["Model"].split(" ")[0]  # "YOLOv8n-det"
        if "train_mode" in state and state["train_mode"] == "finetune":
            pretrained = True
            weights_dst_path = os.path.join(g.app_data_dir, model_filename)
            with urlopen(weights_url) as file:
                weights_size = file.length

            progress = sly.Progress(
                message="",
                total_cnt=weights_size,
                is_size=True,
            )
            progress_cb = partial(download_monitor, g.api=g.api, progress=progress)

            with progress_bar_download_model(
                message="Downloading model weights...",
                total=weights_size,
                unit="bytes",
                unit_scale=True,
            ) as weights_pbar:
                sly.fs.download(
                    url=weights_url,
                    save_path=weights_dst_path,
                    progress=progress_cb,
                )
            model = CustomYOLO(weights_dst_path, stop_event=g.stop_event)
        else:
            yaml_config = selected_dict["yaml_config"]
            pretrained = False
            model = CustomYOLO(yaml_config, stop_event=g.stop_event)
    # elif model_source == "Custom models":
    #     custom_link = model_path_input.get_value()
    #     model_filename = "custom_model.pt"
    #     weights_dst_path = os.path.join(g.app_data_dir, model_filename)
    #     file_info = g.api.file.get_info_by_path(sly.env.team_id(), custom_link)
    #     if file_info is None:
    #         raise FileNotFoundError(f"Custon model file not found: {custom_link}")
    #     file_size = file_info.sizeb
    #     progress = sly.Progress(
    #         message="",
    #         total_cnt=file_size,
    #         is_size=True,
    #     )
    #     progress_cb = partial(download_monitor, g.api=g.api, progress=progress)
    #     with progress_bar_download_model(
    #         message="Downloading model weights...",
    #         total=file_size,
    #         unit="bytes",
    #         unit_scale=True,
    #     ) as weights_pbar:
    #         g.api.file.download(
    #             team_id=sly.env.team_id(),
    #             remote_path=custom_link,
    #             local_save_path=weights_dst_path,
    #             progress_cb=progress_cb,
    #         )
    #     pretrained = True
    #     model = CustomYOLO(weights_dst_path, stop_event=g.stop_event)
    #     try:
    #         # get model_name from previous training
    #         selected_model_name = model.ckpt["sly_metadata"]["model_name"]
    #     except Exception:
    #         selected_model_name = "custom_model.pt"

    model_select_done.show()
    model_not_found_text.hide()
    select_model_button.hide()
    model_tabs.disable()
    models_table.disable()
    model_path_input.disable()
    reselect_model_button.show()
    stepper.set_active_step(4)
    card_train_params.unlock()
    card_train_params.uncollapse()

    # ---------------------------------- Init And Set Workflow Input --------------------------------- #
    w.workflow_input(g.api, g.project_info, file_info)
    # ----------------------------------------------- - ---------------------------------------------- #

    # add callbacks to model
    def on_train_batch_end(trainer):
        with open("train_batches.txt", "w") as file:
            file.write("train batch end")

    def freeze_callback(trainer):
        model = trainer.model
        num_freeze = n_frozen_layers_input.get_value()
        print(f"Freezing {num_freeze} layers...")
        freeze = [f"model.{x}." for x in range(num_freeze)]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f"freezing {k}")
                v.requires_grad = False
        print(f"{num_freeze} layers were frozen")

    model.add_callback("on_train_batch_end", on_train_batch_end)
    if freeze_layers.is_switched():
        model.add_callback("on_train_start", freeze_callback)

    # get additional training params
    additional_params = train_settings_editor.get_text()
    additional_params = yaml.safe_load(additional_params)

    # set up epoch progress bar and grid plot
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    grid_plot_f.show()
    plot_notification.show()

    watch_file = os.path.join(local_artifacts_dir, "results.csv")
    plotted_train_batches = []
    remote_images_path = (
        f"{framework_folder}/{task_type}/{g.project_info.name}/images/{g.app_session_id}/"
    )

    def check_number(value):
        # if value is not str, NaN, infinity or negative infinity
        if isinstance(value, (int, float)) and math.isfinite(value):
            return True
        else:
            return False

    def on_results_file_changed(filepath, pbar):
        # read results file
        results = pd.read_csv(filepath)
        results.columns = [col.replace(" ", "") for col in results.columns]
        print(results.tail(1))
        # get losses values
        train_box_loss = results["train/box_loss"].iat[-1]
        train_cls_loss = results["train/cls_loss"].iat[-1]
        train_dfl_loss = results["train/dfl_loss"].iat[-1]
        if "train/pose_loss" in results.columns:
            train_pose_loss = results["train/pose_loss"].iat[-1]
        if "train/kobj_loss" in results.columns:
            train_kobj_loss = results["train/kobj_loss"].iat[-1]
        if "train/seg_loss" in results.columns:
            train_seg_loss = results["train/seg_loss"].iat[-1]
        precision = results["metrics/precision(B)"].iat[-1]
        recall = results["metrics/recall(B)"].iat[-1]
        val_box_loss = results["val/box_loss"].iat[-1]
        val_cls_loss = results["val/cls_loss"].iat[-1]
        val_dfl_loss = results["val/dfl_loss"].iat[-1]
        if "val/pose_loss" in results.columns:
            val_pose_loss = results["val/pose_loss"].iat[-1]
        if "val/kobj_loss" in results.columns:
            val_kobj_loss = results["val/kobj_loss"].iat[-1]
        if "val/seg_loss" in results.columns:
            val_seg_loss = results["val/seg_loss"].iat[-1]
        # update progress bar
        x = results["epoch"].iat[-1]
        pbar.update(int(x) + 1 - pbar.n)
        # add new points to plots
        if check_number(float(train_box_loss)):
            grid_plot.add_scalar("train/box loss", float(train_box_loss), int(x))
        if check_number(float(train_cls_loss)):
            grid_plot.add_scalar("train/cls loss", float(train_cls_loss), int(x))
        if check_number(float(train_dfl_loss)):
            grid_plot.add_scalar("train/dfl loss", float(train_dfl_loss), int(x))
        if "train/pose_loss" in results.columns:
            if check_number(float(train_pose_loss)):
                grid_plot.add_scalar("train/pose loss", float(train_pose_loss), int(x))
        if "train/kobj_loss" in results.columns:
            if check_number(float(train_kobj_loss)):
                grid_plot.add_scalar("train/kobj loss", float(train_kobj_loss), int(x))
        if "train/seg_loss" in results.columns:
            if check_number(float(train_seg_loss)):
                grid_plot.add_scalar("train/seg loss", float(train_seg_loss), int(x))
        if check_number(float(precision)):
            grid_plot.add_scalar("precision & recall/precision", float(precision), int(x))
        if check_number(float(recall)):
            grid_plot.add_scalar("precision & recall/recall", float(recall), int(x))
        if check_number(float(val_box_loss)):
            grid_plot.add_scalar("val/box loss", float(val_box_loss), int(x))
        if check_number(float(val_cls_loss)):
            grid_plot.add_scalar("val/cls loss", float(val_cls_loss), int(x))
        if check_number(float(val_dfl_loss)):
            grid_plot.add_scalar("val/dfl loss", float(val_dfl_loss), int(x))
        if "val/pose_loss" in results.columns:
            if check_number(float(val_pose_loss)):
                grid_plot.add_scalar("val/pose loss", float(val_pose_loss), int(x))
        if "val/kobj_loss" in results.columns:
            if check_number(float(val_kobj_loss)):
                grid_plot.add_scalar("val/kobj loss", float(val_kobj_loss), int(x))
        if "val/seg_loss" in results.columns:
            if check_number(float(val_seg_loss)):
                grid_plot.add_scalar("val/seg loss", float(val_seg_loss), int(x))
        # visualize train batch
        batch = f"train_batch{x-1}.jpg"
        local_train_batches_path = os.path.join(local_artifacts_dir, batch)
        if (
            os.path.exists(local_train_batches_path)
            and batch not in plotted_train_batches
            and x < 10
        ):
            plotted_train_batches.append(batch)
            remote_train_batches_path = os.path.join(remote_images_path, batch)
            tf_train_batches_info = g.api.file.upload(
                team_id, local_train_batches_path, remote_train_batches_path
            )
            train_batches_gallery.append(tf_train_batches_info.storage_path)
            if x == 1:
                train_batches_gallery_f.show()

    watcher = Watcher(
        watch_file,
        on_results_file_changed,
        progress_bar_epochs(
            message="Epochs:", total=state.get("n_epochs", n_epochs_input.get_value())
        ),
    )
    # train model and upload best checkpoints to team files
    device = 0 if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(g.yolov8_project_dir, "data_config.yaml")
    sly.logger.info(f"Using device: {device}")

    def watcher_func():
        watcher.watch()

    def disable_watcher():
        watcher.running = False

    app.call_before_shutdown(disable_watcher)

    threading.Thread(target=watcher_func, daemon=True).start()
    if len(train_set) > 300:
        n_train_batches = math.ceil(len(train_set) / batch_size_input.get_value())
        train_batches_filepath = "train_batches.txt"

        def on_train_batches_file_changed(filepath, pbar):
            g.train_counter += 1
            if g.train_counter % n_train_batches == 0:
                g.train_counter = 0
                pbar.reset()
            else:
                pbar.update(g.train_counter % n_train_batches - pbar.n)

        train_batch_watcher = Watcher(
            train_batches_filepath,
            on_train_batches_file_changed,
            progress_bar_iters(message="Training batches:", total=n_train_batches),
        )

        def train_batch_watcher_func():
            train_batch_watcher.watch()

        def train_batch_watcher_disable():
            train_batch_watcher.running = False

        app.call_before_shutdown(train_batch_watcher_disable)

        threading.Thread(target=train_batch_watcher_func, daemon=True).start()

    # extract training hyperparameters
    n_epochs = state.get("n_epochs", n_epochs_input.get_value())
    patience = state.get("patience", patience_input.get_value())
    batch_size = state.get("batch_size", batch_size_input.get_value())
    image_size = state.get("input_image_size", image_size_input.get_value())
    n_workers = state.get("n_workers", n_workers_input.get_value())
    optimizer = state.get("optimizer", select_optimizer.get_value())
    lr0 = state.get("lr0", additional_params["lr0"])
    lrf = state.get("lrf", additional_params["lr0"])
    momentum = state.get("momentum", additional_params["momentum"])
    weight_decay = state.get("weight_decay", additional_params["weight_decay"])
    warmup_epochs = state.get("warmup_epochs", additional_params["warmup_epochs"])
    warmup_momentum = state.get("warmup_momentum", additional_params["warmup_momentum"])
    warmup_bias_lr = state.get("warmup_bias_lr", additional_params["warmup_bias_lr"])
    amp = state.get("amp", additional_params["amp"])
    hsv_h = state.get("hsv_h", additional_params["hsv_h"])
    hsv_s = state.get("hsv_s", additional_params["hsv_s"])
    hsv_v = state.get("hsv_v", additional_params["hsv_v"])
    degrees = state.get("degrees", additional_params["degrees"])
    translate = state.get("translate", additional_params["translate"])
    scale = state.get("scale", additional_params["scale"])
    shear = state.get("shear", additional_params["shear"])
    perspective = state.get("perspective", additional_params["perspective"])
    flipud = state.get("flipud", additional_params["flipud"])
    fliplr = state.get("fliplr", additional_params["fliplr"])
    mosaic = state.get("mosaic", additional_params["mosaic"])
    mixup = state.get("mixup", additional_params["mixup"])
    copy_paste = state.get("copy_paste", additional_params["copy_paste"])

    if pretrained:
        select_train_mode.set_value(value="Finetune mode")
    else:
        select_train_mode.set_value(value="Scratch mode")

    n_epochs_input.value = n_epochs
    patience_input.value = patience
    batch_size_input.value = batch_size
    image_size_input.value = image_size
    select_optimizer.set_value(value=optimizer)
    n_workers_input.value = n_workers

    additional_params_text = train_settings_editor.get_text()
    ryaml = ruamel.yaml.YAML()
    additional_params_dict = ryaml.load(additional_params_text)
    additional_params_dict["lr0"] = lr0
    additional_params_dict["lrf"] = lrf
    additional_params_dict["momentum"] = momentum
    additional_params_dict["weight_decay"] = weight_decay
    additional_params_dict["warmup_epochs"] = warmup_epochs
    additional_params_dict["warmup_momentum"] = warmup_momentum
    additional_params_dict["warmup_bias_lr"] = warmup_bias_lr
    additional_params_dict["amp"] = amp
    additional_params_dict["hsv_h"] = hsv_h
    additional_params_dict["hsv_s"] = hsv_s
    additional_params_dict["hsv_v"] = hsv_v
    additional_params_dict["degrees"] = degrees
    additional_params_dict["translate"] = translate
    additional_params_dict["scale"] = scale
    additional_params_dict["shear"] = shear
    additional_params_dict["perspective"] = perspective
    additional_params_dict["flipud"] = flipud
    additional_params_dict["fliplr"] = fliplr
    additional_params_dict["mixup"] = mixup
    additional_params_dict["copy_paste"] = copy_paste
    stream = io.BytesIO()
    ryaml.dump(additional_params_dict, stream)
    additional_params_str = stream.getvalue()
    additional_params_str = additional_params_str.decode("utf-8")
    train_settings_editor.set_text(additional_params_str)

    save_train_params_button.hide()
    train_params_done.show()
    reselect_train_params_button.show()
    select_train_mode.disable()
    n_epochs_input.disable()
    patience_input.disable()
    batch_size_input.disable()
    image_size_input.disable()
    select_optimizer.disable()
    n_workers_input.disable()
    run_model_benchmark_checkbox.disable()
    run_speedtest_checkbox.disable()
    export_model_switch.disable()
    export_onnx_checkbox.disable()
    export_tensorrt_checkbox.disable()
    train_settings_editor.readonly = True
    stepper.set_active_step(5)
    card_train_progress.unlock()
    card_train_progress.uncollapse()

    def train_model():
        model.train(
            data=data_path,
            project=checkpoint_dir,
            epochs=n_epochs,
            patience=patience,
            batch=batch_size,
            imgsz=image_size,
            save_period=1000,
            device=device,
            workers=n_workers,
            optimizer=optimizer,
            pretrained=pretrained,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            amp=amp,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
        )

    stop_training_tooltip.show()

    train_thread = threading.Thread(target=train_model, args=())
    train_thread.start()
    train_thread.join()
    watcher.running = False
    progress_bar_iters.hide()
    progress_bar_epochs.hide()

    # visualize model predictions
    making_training_vis_f.show()
    # visualize model predictions
    for i in range(4):
        val_batch_labels_id, val_batch_preds_id = None, None
        labels_path = os.path.join(local_artifacts_dir, f"val_batch{i}_labels.jpg")
        if os.path.exists(labels_path):
            remote_labels_path = os.path.join(remote_images_path, f"val_batch{i}_labels.jpg")
            tf_labels_info = g.api.file.upload(team_id, labels_path, remote_labels_path)
            val_batch_labels_id = val_batches_gallery.append(
                image_url=tf_labels_info.storage_path,
                title="labels",
            )
        preds_path = os.path.join(local_artifacts_dir, f"val_batch{i}_pred.jpg")
        if os.path.exists(preds_path):
            remote_preds_path = os.path.join(remote_images_path, f"val_batch{i}_pred.jpg")
            tf_preds_info = g.api.file.upload(team_id, preds_path, remote_preds_path)
            val_batch_preds_id = val_batches_gallery.append(
                image_url=tf_preds_info.storage_path,
                title="predictions",
            )
        if val_batch_labels_id and val_batch_preds_id:
            val_batches_gallery.sync_images([val_batch_labels_id, val_batch_preds_id])
        if i == 0:
            val_batches_gallery_f.show()

    stop_training_tooltip.loading = False
    stop_training_tooltip.hide()

    # visualize additional training results
    confusion_matrix_path = os.path.join(local_artifacts_dir, "confusion_matrix_normalized.png")
    if os.path.exists(confusion_matrix_path):
        remote_confusion_matrix_path = os.path.join(
            remote_images_path, "confusion_matrix_normalized.png"
        )
        tf_confusion_matrix_info = g.api.file.upload(
            team_id, confusion_matrix_path, remote_confusion_matrix_path
        )
        if not app.is_stopped():
            additional_gallery.append(tf_confusion_matrix_info.storage_path)
            additional_gallery_f.show()
    pr_curve_path = os.path.join(local_artifacts_dir, "PR_curve.png")
    if os.path.exists(pr_curve_path):
        remote_pr_curve_path = os.path.join(remote_images_path, "PR_curve.png")
        tf_pr_curve_info = g.api.file.upload(team_id, pr_curve_path, remote_pr_curve_path)
        if not app.is_stopped():
            additional_gallery.append(tf_pr_curve_info.storage_path)
    f1_curve_path = os.path.join(local_artifacts_dir, "F1_curve.png")
    if os.path.exists(f1_curve_path):
        remote_f1_curve_path = os.path.join(remote_images_path, "F1_curve.png")
        tf_f1_curve_info = g.api.file.upload(team_id, f1_curve_path, remote_f1_curve_path)
        if not app.is_stopped():
            additional_gallery.append(tf_f1_curve_info.storage_path)
    box_f1_curve_path = os.path.join(local_artifacts_dir, "BoxF1_curve.png")
    if os.path.exists(box_f1_curve_path):
        remote_box_f1_curve_path = os.path.join(remote_images_path, "BoxF1_curve.png")
        tf_box_f1_curve_info = g.api.file.upload(team_id, box_f1_curve_path, remote_box_f1_curve_path)
        if not app.is_stopped():
            additional_gallery.append(tf_box_f1_curve_info.storage_path)
    pose_f1_curve_path = os.path.join(local_artifacts_dir, "PoseF1_curve.png")
    if os.path.exists(pose_f1_curve_path):
        remote_pose_f1_curve_path = os.path.join(remote_images_path, "PoseF1_curve.png")
        tf_pose_f1_curve_info = g.api.file.upload(
            team_id, pose_f1_curve_path, remote_pose_f1_curve_path
        )
        if not app.is_stopped():
            additional_gallery.append(tf_pose_f1_curve_info.storage_path)
    mask_f1_curve_path = os.path.join(local_artifacts_dir, "MaskF1_curve.png")
    if os.path.exists(mask_f1_curve_path):
        remote_mask_f1_curve_path = os.path.join(remote_images_path, "MaskF1_curve.png")
        tf_mask_f1_curve_info = g.api.file.upload(
            team_id, mask_f1_curve_path, remote_mask_f1_curve_path
        )
        if not app.is_stopped():
            additional_gallery.append(tf_mask_f1_curve_info.storage_path)

    making_training_vis_f.hide()

    # rename best checkpoint file
    if not os.path.isfile(watch_file):
        sly.logger.warning(
            "The file with results does not exist, training was not completed successfully."
        )
        app.stop()
        return
    results = pd.read_csv(watch_file)
    results.columns = [col.replace(" ", "") for col in results.columns]
    results["fitness"] = (0.1 * results["metrics/mAP50(B)"]) + (
        0.9 * results["metrics/mAP50-95(B)"]
    )
    print("Final results:")
    print(results)
    best_epoch = results["fitness"].idxmax()
    best_filename = f"best_{best_epoch}.pt"
    current_best_filepath = os.path.join(local_artifacts_dir, "weights", "best.pt")
    new_best_filepath = os.path.join(local_artifacts_dir, "weights", best_filename)
    os.rename(current_best_filepath, new_best_filepath)

    # add model name to saved weights
    def add_sly_metadata_to_ckpt(ckpt_path):
        loaded = torch.load(ckpt_path, map_location="cpu")
        loaded["sly_metadata"] = {"model_name": selected_model_name}
        torch.save(loaded, ckpt_path)

    best_path = os.path.join(local_artifacts_dir, "weights", best_filename)
    last_path = os.path.join(local_artifacts_dir, "weights", "last.pt")
    if os.path.exists(best_path):
        add_sly_metadata_to_ckpt(best_path)
    if os.path.exists(last_path):
        add_sly_metadata_to_ckpt(last_path)

    # save link to app ui
    app_url = f"/apps/sessions/{g.app_session_id}"
    app_link_path = os.path.join(local_artifacts_dir, "open_app.lnk")
    with open(app_link_path, "w") as text_file:
        print(app_url, file=text_file)

    # Exporting to ONNX / TensorRT
    if export_model_switch.is_switched() and os.path.exists(best_path):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            export_weights(best_path, selected_model_name, model_benchmark_pbar)
        except Exception as e:
            sly.logger.error(f"Error during model export: {e}")
        finally:
            model_benchmark_pbar.hide()

    # upload training artifacts to team files
    upload_artifacts_dir = os.path.join(
        framework_folder,
        task_type_select.get_value(),
        g.project_info.name,
        str(g.app_session_id),
    )

    if not app.is_stopped():

        def upload_monitor(monitor, g.api: sly.Api, progress: sly.Progress):
            value = monitor.bytes_read
            if progress.total == 0:
                progress.set(value, monitor.len, report=False)
            else:
                progress.set_current_value(value, report=False)
            artifacts_pbar.update(progress.current - artifacts_pbar.n)

        local_files = sly.fs.list_files_recursively(local_artifacts_dir)
        total_size = sum([sly.fs.get_file_size(file_path) for file_path in local_files])
        progress = sly.Progress(
            message="",
            total_cnt=total_size,
            is_size=True,
        )
        progress_cb = partial(upload_monitor, g.api=g.api, progress=progress)
        with progress_bar_upload_artifacts(
            message="Uploading train artifacts to Team Files...",
            total=total_size,
            unit="bytes",
            unit_scale=True,
        ) as artifacts_pbar:
            remote_artifacts_dir = g.api.file.upload_directory(
                team_id=sly.env.team_id(),
                local_dir=local_artifacts_dir,
                remote_dir=upload_artifacts_dir,
                progress_size_cb=progress_cb,
            )
        progress_bar_upload_artifacts.hide()
    else:
        sly.logger.info(
            "Uploading training artifacts before stopping the app... (progress bar is disabled)"
        )
        remote_artifacts_dir = g.api.file.upload_directory(
            team_id=sly.env.team_id(),
            local_dir=local_artifacts_dir,
            remote_dir=upload_artifacts_dir,
        )
        sly.logger.info("Training artifacts uploaded successfully")
    remote_weights_dir = yolov8_artifacts.get_weights_path(remote_artifacts_dir)

    # ------------------------------------- Model Benchmark ------------------------------------- #
    model_benchmark_done = False
    if run_model_benchmark_checkbox.is_checked():
        try:
            if task_type in [TaskType.INSTANCE_SEGMENTATION, TaskType.OBJECT_DETECTION]:
                sly.logger.info(f"Creating the report for the best model: {best_filename!r}")
                creating_report_f.show()
                model_benchmark_pbar.show()
                model_benchmark_pbar(message="Starting Model Benchmark evaluation...", total=1)

                # 0. Serve trained model
                m = YOLOv8ModelMB(
                    model_dir=local_artifacts_dir + "/weights",
                    use_gui=False,
                    custom_inference_settings=os.path.join(
                        root_source_path, "serve", "custom_settings.yaml"
                    ),
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                sly.logger.info(f"Using device: {device}")

                checkpoint_path = os.path.join(remote_weights_dir, best_filename)
                deploy_params = dict(
                    device=device,
                    runtime=sly.nn.inference.RuntimeType.PYTORCH,
                    model_source="Custom models",
                    task_type=task_type,
                    checkpoint_name=best_filename,
                    checkpoint_url=checkpoint_path,
                )
                m._load_model(deploy_params)
                m.serve()
                m.model.overrides["verbose"] = False
                session = SessionJSON(g.api, session_url="http://localhost:8000")
                sly.fs.remove_dir(g.app_data_dir + "/benchmark")

                # 1. Init benchmark (todo: auto-detect task type)
                benchmark_dataset_ids = None
                benchmark_images_ids = None
                train_dataset_ids = None
                train_images_ids = None

                split_method = train_val_split._content.get_active_tab()

                if split_method == "Based on datasets":
                    benchmark_dataset_ids = train_val_split._val_ds_select.get_selected_ids()
                    train_dataset_ids = train_val_split._train_ds_select.get_selected_ids()
                else:

                    def get_image_infos_by_split(split: list):
                        ds_infos_dict = {ds_info.name: ds_info for ds_info in dataset_infos}
                        image_names_per_dataset = {}
                        for item in split:
                            image_names_per_dataset.setdefault(item.dataset_name, []).append(
                                item.name
                            )
                        image_infos = []
                        for (
                            dataset_name,
                            image_names,
                        ) in image_names_per_dataset.items():
                            ds_info = ds_infos_dict[dataset_name]
                            image_infos.extend(
                                g.api.image.get_list(
                                    ds_info.id,
                                    filters=[
                                        {
                                            "field": "name",
                                            "operator": "in",
                                            "value": image_names,
                                        }
                                    ],
                                )
                            )
                        return image_infos

                    val_image_infos = get_image_infos_by_split(val_set)
                    train_image_infos = get_image_infos_by_split(train_set)
                    benchmark_images_ids = [img_info.id for img_info in val_image_infos]
                    train_images_ids = [img_info.id for img_info in train_image_infos]

                if task_type == TaskType.OBJECT_DETECTION:
                    bm = ObjectDetectionBenchmark(
                        g.api,
                        g.project_info.id,
                        output_dir=g.app_data_dir + "/benchmark",
                        gt_dataset_ids=benchmark_dataset_ids,
                        gt_images_ids=benchmark_images_ids,
                        progress=model_benchmark_pbar,
                        progress_secondary=model_benchmark_pbar_secondary,
                        classes_whitelist=selected_classes,
                    )
                elif task_type == TaskType.INSTANCE_SEGMENTATION:
                    bm = InstanceSegmentationBenchmark(
                        g.api,
                        g.project_info.id,
                        output_dir=g.app_data_dir + "/benchmark",
                        gt_dataset_ids=benchmark_dataset_ids,
                        gt_images_ids=benchmark_images_ids,
                        progress=model_benchmark_pbar,
                        progress_secondary=model_benchmark_pbar_secondary,
                        classes_whitelist=selected_classes,
                    )
                else:
                    raise ValueError(
                        f"Model benchmark for task type {task_type} is not implemented (coming soon)"
                    )

                train_info = {
                    "app_session_id": g.app_session_id,
                    "train_dataset_ids": train_dataset_ids,
                    "train_images_ids": train_images_ids,
                    "images_count": len(train_set),
                }
                bm.train_info = train_info

                # 2. Run inference
                bm.run_inference(session)

                # 3. Pull results from the server
                gt_project_path, dt_project_path = bm.download_projects(save_images=False)

                # 4. Evaluate
                bm._evaluate(gt_project_path, dt_project_path)

                # 5. Upload evaluation results
                eval_res_dir = get_eval_results_dir_name(g.api, g.app_session_id, g.project_info)
                bm.upload_eval_results(eval_res_dir + "/evaluation/")

                # 6. Speed test
                if run_speedtest_checkbox.is_checked():
                    bm.run_speedtest(session, g.project_info.id)
                    model_benchmark_pbar_secondary.hide()
                    bm.upload_speedtest_results(eval_res_dir + "/speedtest/")

                # 7. Prepare visualizations, report and upload
                bm.visualize()
                remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
                report = bm.upload_report_link(remote_dir)

                # 8. UI updates
                benchmark_report_template = g.api.file.get_info_by_path(
                    sly.env.team_id(), remote_dir + "template.vue"
                )
                model_benchmark_done = True
                creating_report_f.hide()
                model_benchmark_report.set(benchmark_report_template)
                model_benchmark_report.show()
                model_benchmark_pbar.hide()
                sly.logger.info(
                    f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
                )
                sly.logger.info(
                    f"Differences project name: {bm.diff_project_info.name}. Workspace_id: {bm.diff_project_info.workspace_id}"
                )
        except Exception as e:
            sly.logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
            creating_report_f.hide()
            model_benchmark_pbar.hide()
            model_benchmark_pbar_secondary.hide()
            try:
                if bm.dt_project_info:
                    g.api.project.remove(bm.dt_project_info.id)
                if bm.diff_project_info:
                    g.api.project.remove(bm.diff_project_info.id)
            except Exception as e2:
                pass
    # ----------------------------------------------- - ---------------------------------------------- #

    # ------------------------------------- Set Workflow Outputs ------------------------------------- #
    if not model_benchmark_done:
        benchmark_report_template = None
    w.workflow_output(
        g.api,
        model_filename,
        remote_artifacts_dir,
        best_filename,
        benchmark_report_template,
    )
    # ----------------------------------------------- - ---------------------------------------------- #

    if not app.is_stopped():
        file_info = g.api.file.get_info_by_path(
            sly.env.team_id(), remote_artifacts_dir + "/results.csv"
        )
        train_artifacts_folder.set(file_info)
        # finish training
        card_train_artifacts.unlock()
        card_train_artifacts.uncollapse()

    # upload sly_metadata.json
    yolov8_artifacts.generate_metadata(
        app_name=yolov8_artifacts.app_name,
        task_id=g.app_session_id,
        artifacts_folder=remote_artifacts_dir,
        weights_folder=remote_weights_dir,
        weights_ext=yolov8_artifacts.weights_ext,
        project_name=g.project_info.name,
        task_type=task_type,
        config_path=None,
    )

    # delete app data since it is no longer needed
    sly.fs.remove_dir(g.app_data_dir)
    sly.fs.silent_remove("train_batches.txt")
    # set task output
    sly.output.set_directory(remote_artifacts_dir)
    # stop app
    app.stop()
    return {"result": "successfully finished automatic training session"}


def export_weights(weights_path, selected_model_name, progress: SlyTqdm):
    from src.model_export import export_checkpoint

    checkpoint_info_path = dump_yaml_checkpoint_info(weights_path, selected_model_name)
    pbar = None
    fp16 = export_fp16_switch.is_switched()
    if export_tensorrt_checkbox.is_checked():
        pbar = progress(message="Exporting model to TensorRT, this may take some time...", total=1)
        export_checkpoint(weights_path, format="engine", fp16=fp16, dynamic=False)
        pbar.update(1)
    if export_onnx_checkbox.is_checked():
        pbar = progress(message="Exporting model to ONNX...", total=1)
        dynamic = not fp16  # dynamic mode is not supported for fp16
        export_checkpoint(weights_path, format="onnx", fp16=fp16, dynamic=dynamic)
        pbar.update(1)


def dump_yaml_checkpoint_info(weights_path, selected_model_name):
    p = r"yolov(\d+)"
    match = re.match(p, selected_model_name.lower())
    architecture = match.group(0) if match else None
    checkpoint_info = {
        "model_name": selected_model_name,
        "architecture": architecture,
    }
    checkpoint_info_path = os.path.join(os.path.dirname(weights_path), "checkpoint_info.yaml")
    with open(checkpoint_info_path, "w") as f:
        yaml.dump(checkpoint_info, f)
    return checkpoint_info_path
