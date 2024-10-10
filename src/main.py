import os
import src.sly_globals as g
from fastapi import Request
import supervisely as sly
from supervisely.app.widgets import Container
import src.workflow as w
# import src.ui.input_project as input_project
# import src.ui.task as task
# import src.ui.models as models
# import src.ui.classes as classes
# import src.ui.train_val_split as train_val_split
# import src.ui.graphics as graphics
# import src.ui.hyperparameters as hyperparameters
import src.ui.handlers as handlers

# import src.ui.train as train
# import src.ui.augmentations as augmentations
# import src.ui.model_leaderboard as model_leaderboard


# widgets = [
#     input_project.card,
#     Container(widgets=[task.card, model_leaderboard.card]),
#     models.card,
#     classes.card,
#     train_val_split.card,
#     augmentations.card,
#     hyperparameters.card,
#     train.card,
# ]

# stepper = Stepper(widgets=widgets)

layout = Container(widgets=[handlers.stepper])
app = sly.Application(layout=layout)

g.app = app


server = g.app.get_server()

@server.post("/auto_train")
def auto_train(request: Request):
    sly.logger.info("Starting automatic training session...")
    state = request.state.state
    project_id = state["project_id"]
    task_type = state["task_type"]
    use_cache = state.get("use_cache", True)

    local_dir = g.root_model_checkpoint_dir
    if task_type == "object detection":
        necessary_geometries = ["rectangle"]
        checkpoint_dir = os.path.join(local_dir, "detect")
        local_artifacts_dir = os.path.join(local_dir, "detect", "train")
        models_data = g.det_models_data
    elif task_type == "pose estimation":
        necessary_geometries = ["graph", "rectangle"]
        checkpoint_dir = os.path.join(local_dir, "pose")
        local_artifacts_dir = os.path.join(local_dir, "pose", "train")
        models_data = g.pose_models_data
    elif task_type == "instance segmentation":
        necessary_geometries = ["bitmap", "polygon"]
        checkpoint_dir = os.path.join(local_dir, "segment")
        local_artifacts_dir = os.path.join(local_dir, "segment", "train")
        models_data = g.seg_models_data

    sly.logger.info(f"Local artifacts dir: {local_artifacts_dir}")

    if os.path.exists(local_artifacts_dir):
        sly.fs.remove_dir(local_artifacts_dir)

    # get number of images in selected datasets
    if "dataset_ids" not in state:
        dataset_infos = api.dataset.get_list(project_id)
        dataset_ids = [dataset_info.id for dataset_info in dataset_infos]
    else:
        dataset_ids = state["dataset_ids"]
        dataset_infos = [
            api.dataset.get_info_by_id(dataset_id) for dataset_id in dataset_ids
        ]

    n_images = sum([info.images_count for info in dataset_infos])
    download_project(
        api=api,
        project_info=project_info,
        dataset_infos=dataset_infos,
        use_cache=use_cache,
        progress=progress_bar_download_project,
    )

    # remove unselected classes
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    selected_classes = [cls.name for cls in project_meta.obj_classes]

    # remove classes with unnecessary shapes
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
                g.project_dir, classes_to_remove=unnecessary_classes, inplace=True
            )
    # extract geometry configs
    if task_type == "pose estimation":
        nodes_order = []
        cls2config = {}
        total_config = {"nodes": {}, "edges": []}
        for cls in project_meta.obj_classes:
            if (
                cls.name in selected_classes
                and cls.geometry_type.geometry_name() == "graph"
            ):
                g.keypoints_classes.append(cls.name)
                geometry_config = cls.geometry_config
                cls2config[cls.name] = geometry_config
                for key, value in geometry_config["nodes"].items():
                    label = value["label"]
                    g.node_id2label[key] = label
                    if label not in total_config["nodes"]:
                        total_config["nodes"][label] = value
                        nodes_order.append(label)
        if len(total_config["nodes"]) == 17:
            total_config["nodes"][uuid.uuid4().hex[:6]] = {
                "label": "fictive",
                "color": [0, 0, 255],
                "loc": [0, 0],
            }
        g.keypoints_template = total_config

    # transfer project to detection task if necessary
    if task_type == "object detection":
        sly.Project.to_detection_task(g.project_dir, inplace=True)
    # split the data
    try:
        train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        train_set, val_set = train_val_split.get_splits()
    except Exception:
        if not use_cache:
            raise
        sly.logger.warning(
            "Error during data splitting. Will try to re-download project without cache",
            exc_info=True,
        )
        download_project(
            api=api,
            project_info=project_info,
            dataset_infos=dataset_infos,
            use_cache=False,
            progress=progress_bar_download_project,
        )
        train_val_split._project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        train_set, val_set = train_val_split.get_splits()
    verify_train_val_sets(train_set, val_set)
    # convert dataset from supervisely to yolo format
    if os.path.exists(g.yolov8_project_dir):
        sly.fs.clean_dir(g.yolov8_project_dir)
    transform(
        g.project_dir,
        g.yolov8_project_dir,
        train_set,
        val_set,
        progress_bar_convert_to_yolo,
        task_type,
    )
    # download model
    weights_type = "Pretrained models"

    def download_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        weights_pbar.update(progress.current)

    file_info = None

    g.stop_event = threading.Event()

    if weights_type == "Pretrained models":
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
        if "train_mode" in state and state["train_mode"] == "Finetune mode":
            pretrained = True
            weights_dst_path = os.path.join(g.app_data_dir, model_filename)
            with urlopen(weights_url) as file:
                weights_size = file.length

            progress = sly.Progress(
                message="",
                total_cnt=weights_size,
                is_size=True,
            )
            progress_cb = partial(download_monitor, api=api, progress=progress)

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
    # elif weights_type == "Custom models":
    #     custom_link = model_path_input.get_value()
    #     model_filename = "custom_model.pt"
    #     weights_dst_path = os.path.join(g.app_data_dir, model_filename)
    #     file_info = api.file.get_info_by_path(sly.env.team_id(), custom_link)
    #     if file_info is None:
    #         raise FileNotFoundError(f"Custon model file not found: {custom_link}")
    #     file_size = file_info.sizeb
    #     progress = sly.Progress(
    #         message="",
    #         total_cnt=file_size,
    #         is_size=True,
    #     )
    #     progress_cb = partial(download_monitor, api=api, progress=progress)
    #     with progress_bar_download_model(
    #         message="Downloading model weights...",
    #         total=file_size,
    #         unit="bytes",
    #         unit_scale=True,
    #     ) as weights_pbar:
    #         api.file.download(
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

    # ---------------------------------- Init And Set Workflow Input --------------------------------- #
    w.workflow_input(api, project_info, file_info)
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
    if task_type == "pose estimation":
        additional_params["fliplr"] = 0.0
    # set up epoch progress bar and grid plot
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    watch_file = os.path.join(local_artifacts_dir, "results.csv")
    plotted_train_batches = []
    remote_images_path = (
        f"{framework_folder}/{task_type}/{project_info.name}/images/{g.app_session_id}/"
    )

    def check_number(value):
        # if value is not str, NaN, infinity or negative infinity
        if isinstance(value, (int, float)) and math.isfinite(value):
            return True
        else:
            return False

    def on_results_file_changed(filepath, pbar):
        results = pd.read_csv(filepath)
        results.columns = [col.replace(" ", "") for col in results.columns]
        print(results.tail(1))
        x = results["epoch"].iat[-1]
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
            tf_train_batches_info = api.file.upload(
                team_id, local_train_batches_path, remote_train_batches_path
            )

    watcher = Watcher(
        watch_file,
        on_results_file_changed,
        progress_bar_epochs(message="Epochs:", total=n_epochs_input.get_value()),
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

    def train_model():
        model.train(
            data=data_path,
            project=checkpoint_dir,
            epochs=state.get("n_epochs", n_epochs_input.get_value()),
            patience=state.get("patience", patience_input.get_value()),
            batch=state.get("batch_size", batch_size_input.get_value()),
            imgsz=state.get("input_image_size", image_size_input.get_value()),
            save_period=1000,
            device=device,
            workers=state.get("n_workers", n_workers_input.get_value()),
            optimizer=state.get("optimizer", select_optimizer.get_value()),
            pretrained=pretrained,
            lr0=state.get("lr0", additional_params["lr0"]),
            lrf=state.get("lrf", additional_params["lr0"]),
            momentum=state.get("momentum", additional_params["momentum"]),
            weight_decay=state.get("weight_decay", additional_params["weight_decay"]),
            warmup_epochs=state.get(
                "warmup_epochs", additional_params["warmup_epochs"]
            ),
            warmup_momentum=state.get(
                "warmup_momentum", additional_params["warmup_momentum"]
            ),
            warmup_bias_lr=state.get(
                "warmup_bias_lr", additional_params["warmup_bias_lr"]
            ),
            amp=state.get("amp", additional_params["amp"]),
            hsv_h=state.get("hsv_h", additional_params["hsv_h"]),
            hsv_s=state.get("hsv_s", additional_params["hsv_s"]),
            hsv_v=state.get("hsv_v", additional_params["hsv_v"]),
            degrees=state.get("degrees", additional_params["degrees"]),
            translate=state.get("translate", additional_params["translate"]),
            scale=state.get("scale", additional_params["scale"]),
            shear=state.get("shear", additional_params["shear"]),
            perspective=state.get("perspective", additional_params["perspective"]),
            flipud=state.get("flipud", additional_params["flipud"]),
            fliplr=state.get("fliplr", additional_params["fliplr"]),
            mosaic=state.get("mosaic", additional_params["mosaic"]),
            mixup=state.get("mixup", additional_params["mixup"]),
            copy_paste=state.get("copy_paste", additional_params["copy_paste"]),
        )

    stop_training_tooltip.show()

    train_thread = threading.Thread(target=train_model, args=())
    train_thread.start()
    train_thread.join()
    watcher.running = False

    # visualize model predictions
    for i in range(4):
        val_batch_labels_id, val_batch_preds_id = None, None
        labels_path = os.path.join(local_artifacts_dir, f"val_batch{i}_labels.jpg")
        if os.path.exists(labels_path):
            remote_labels_path = os.path.join(
                remote_images_path, f"val_batch{i}_labels.jpg"
            )
            tf_labels_info = api.file.upload(team_id, labels_path, remote_labels_path)
        preds_path = os.path.join(local_artifacts_dir, f"val_batch{i}_pred.jpg")
        if os.path.exists(preds_path):
            remote_preds_path = os.path.join(
                remote_images_path, f"val_batch{i}_pred.jpg"
            )
            tf_preds_info = api.file.upload(team_id, preds_path, remote_preds_path)

    # visualize additional training results
    confusion_matrix_path = os.path.join(
        local_artifacts_dir, "confusion_matrix_normalized.png"
    )
    if os.path.exists(confusion_matrix_path):
        remote_confusion_matrix_path = os.path.join(
            remote_images_path, "confusion_matrix_normalized.png"
        )
        tf_confusion_matrix_info = api.file.upload(
            team_id, confusion_matrix_path, remote_confusion_matrix_path
        )
    pr_curve_path = os.path.join(local_artifacts_dir, "PR_curve.png")
    if os.path.exists(pr_curve_path):
        remote_pr_curve_path = os.path.join(remote_images_path, "PR_curve.png")
        tf_pr_curve_info = api.file.upload(team_id, pr_curve_path, remote_pr_curve_path)
    f1_curve_path = os.path.join(local_artifacts_dir, "F1_curve.png")
    if os.path.exists(f1_curve_path):
        remote_f1_curve_path = os.path.join(remote_images_path, "F1_curve.png")
        tf_f1_curve_info = api.file.upload(team_id, f1_curve_path, remote_f1_curve_path)
    box_f1_curve_path = os.path.join(local_artifacts_dir, "BoxF1_curve.png")
    if os.path.exists(box_f1_curve_path):
        remote_box_f1_curve_path = os.path.join(remote_images_path, "BoxF1_curve.png")
        tf_box_f1_curve_info = api.file.upload(
            team_id, box_f1_curve_path, remote_box_f1_curve_path
        )
    pose_f1_curve_path = os.path.join(local_artifacts_dir, "PoseF1_curve.png")
    if os.path.exists(pose_f1_curve_path):
        remote_pose_f1_curve_path = os.path.join(remote_images_path, "PoseF1_curve.png")
        tf_pose_f1_curve_info = api.file.upload(
            team_id, pose_f1_curve_path, remote_pose_f1_curve_path
        )
    mask_f1_curve_path = os.path.join(local_artifacts_dir, "MaskF1_curve.png")
    if os.path.exists(mask_f1_curve_path):
        remote_mask_f1_curve_path = os.path.join(remote_images_path, "MaskF1_curve.png")
        tf_mask_f1_curve_info = api.file.upload(
            team_id, mask_f1_curve_path, remote_mask_f1_curve_path
        )

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

    # add geometry config to saved weights for pose estimation task
    if task_type == "pose estimation":
        weights_filepath = os.path.join(local_artifacts_dir, "weights", best_filename)
        weights_dict = torch.load(weights_filepath)
        if len(cls2config.keys()) == 1:
            geometry_config = list(cls2config.values())[0]
            weights_dict["geometry_config"] = geometry_config
        elif len(cls2config.keys()) > 1:
            weights_dict["geometry_config"] = {
                "configs": cls2config,
                "nodes_order": nodes_order,
            }
        torch.save(weights_dict, weights_filepath)

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
        project_info.name,
        str(g.app_session_id),
    )

    if not app.is_stopped():

        def upload_monitor(monitor, api: sly.Api, progress: sly.Progress):
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
        progress_cb = partial(upload_monitor, api=api, progress=progress)
        with progress_bar_upload_artifacts(
            message="Uploading train artifacts to Team Files...",
            total=total_size,
            unit="bytes",
            unit_scale=True,
        ) as artifacts_pbar:
            remote_artifacts_dir = api.file.upload_directory(
                team_id=sly.env.team_id(),
                local_dir=local_artifacts_dir,
                remote_dir=upload_artifacts_dir,
                progress_size_cb=progress_cb,
            )
    else:
        sly.logger.info(
            "Uploading training artifacts before stopping the app... (progress bar is disabled)"
        )
        remote_artifacts_dir = api.file.upload_directory(
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
                sly.logger.info(
                    f"Creating the report for the best model: {best_filename!r}"
                )
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
                m.model.overrides['verbose'] = False
                session = SessionJSON(api, session_url="http://localhost:8000")
                sly.fs.remove_dir(g.app_data_dir + "/benchmark")

                # 1. Init benchmark (todo: auto-detect task type)
                benchmark_dataset_ids = None
                benchmark_images_ids = None
                train_dataset_ids = None
                train_images_ids = None

                split_method = train_val_split._content.get_active_tab()

                if split_method == "Based on datasets":
                    benchmark_dataset_ids = (
                        train_val_split._val_ds_select.get_selected_ids()
                    )
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
                        for dataset_name, image_names in image_names_per_dataset.items():
                            ds_info = ds_infos_dict[dataset_name]
                            image_infos.extend(
                                api.image.get_list(
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
                        api,
                        project_info.id,
                        output_dir=g.app_data_dir + "/benchmark",
                        gt_dataset_ids=benchmark_dataset_ids,
                        gt_images_ids=benchmark_images_ids,
                        progress=model_benchmark_pbar,
                        progress_secondary=model_benchmark_pbar_secondary,
                        classes_whitelist=selected_classes,
                    )
                elif task_type == TaskType.INSTANCE_SEGMENTATION:
                    bm = InstanceSegmentationBenchmark(
                        api,
                        project_info.id,
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
                eval_res_dir = get_eval_results_dir_name(
                    api, g.app_session_id, project_info
                )
                bm.upload_eval_results(eval_res_dir + "/evaluation/")

                # 6. Speed test
                if run_speedtest_checkbox.is_checked():
                    bm.run_speedtest(session, project_info.id)
                    model_benchmark_pbar_secondary.hide()
                    bm.upload_speedtest_results(eval_res_dir + "/speedtest/")

                # 7. Prepare visualizations, report and upload
                bm.visualize()
                remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
                report = bm.upload_report_link(remote_dir)

                # 8. UI updates
                benchmark_report_template = api.file.get_info_by_path(
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
                    api.project.remove(bm.dt_project_info.id)
                if bm.diff_project_info:
                    api.project.remove(bm.diff_project_info.id)
            except Exception as e2:
                pass
    # ----------------------------------------------- - ---------------------------------------------- #

    # ------------------------------------- Set Workflow Outputs ------------------------------------- #
    if not model_benchmark_done:
        benchmark_report_template = None
    w.workflow_output(
        api, model_filename, remote_artifacts_dir, best_filename, benchmark_report_template
    )
    # ----------------------------------------------- - ---------------------------------------------- #

    if not app.is_stopped():
        file_info = api.file.get_info_by_path(
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
        project_name=project_info.name,
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