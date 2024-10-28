import os

import sly_globals as g

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import yaml

import src.ui.augmentations as augmentations_ui
import src.ui.classes as classes_ui
import src.ui.graphics as graphics_ui
import src.ui.handlers as handlers_ui
import src.ui.hyperparameters as hyperparameters_ui
import src.ui.input_project as input_project_ui
import src.ui.model_leaderboard as model_leaderboard_ui
import src.ui.models as models_ui
import src.ui.task as task_ui
import src.ui.train as train_ui
import src.ui.train_val_split as train_val_split_ui
import supervisely as sly


def start_auto_train(state: dict):
    if "yaml_string" in state:
        state = yaml.safe_load(state["yaml_string"])

    # Step 1. Input project
    project_id = state["project_id"]
    use_cache = state.get("use_cache", True)

    input_project_ui.project_th.set(g.project_info)

    if use_cache is True:
        input_project_ui.use_cache_checkbox.check()
    else:
        input_project_ui.use_cache_checkbox.uncheck()
    handlers_ui.stepper.set_active_step(2)
    # ----------------------------------------------------------------------------------------------- #

    # Step 2. Select task type
    task_type = state["task_type"]
    task_ui.task_selector.set_value(task_type)
    models_ui.update_architecture(task_ui.task_selector.get_value())
    handlers_ui.stepper.set_active_step(3)
    # ----------------------------------------------------------------------------------------------- #

    # Step 2. Model Leaderboard
    model_leaderboard_ui.update_table(models_ui.models_meta, task_type)
    # ----------------------------------------------------------------------------------------------- #

    # Step 3. Select arch and model
    arch_type = state["arch_type"]
    model_source = state["model_source"]
    model_name = state["model_name"]
    train_mode = state["train_mode"]

    models_ui.arch_select.set_value(arch_type)
    models_ui.radio_tabs.set_active_tab(model_source)
    table_data = models_ui.table.get_json_data()
    model_idx = 0
    for idx, data in enumerate(table_data["raw_rows_data"]):
        if data[0] == model_name:
            model_idx = idx
            break
        else:
            raise ValueError(f"Model {model_name} not found in the table")
    models_ui.table.select_row(model_idx)

    if train_mode == "finetune":
        models_ui.load_from.on()
    else:
        models_ui.load_from.off()
    handlers_ui.stepper.set_active_step(4)
    # ----------------------------------------------------------------------------------------------- #

    # Step 4. Select classes
    # selected_classes = [cls.name for cls in project_meta.obj_classes]
    # n_classes = len(classes_ui.classes.get_selected_classes())
    # if n_classes > 1:
    #     sly.logger.info(f"{n_classes} classes were selected successfully")
    # else:
    #     sly.logger.info(f"{n_classes} class was selected successfully")
    handlers_ui.stepper.set_active_step(5)
    # ----------------------------------------------------------------------------------------------- #

    # Step 5. Split the data
    # try:
    #     train_val_split_ui.splits._project_fs = sly.Project(project_dir, sly.OpenMode.READ)
    #     train_set, val_set = train_val_split_ui.splits.get_splits()
    # except Exception:
    #     if not use_cache:
    #         raise
    #     sly.logger.warning(
    #         "Error during data splitting. Will try to re-download project without cache",
    #         exc_info=True,
    #     )
    #     download_project(
    #         api=g.api,
    #         project_id=g.project_info.id,
    #         project_dir=project_dir,
    #         use_cache=False,
    #         progress=progress_bar_download_project,
    #     )
    #     train_val_split_ui.splits._project_fs = sly.Project(project_dir, sly.OpenMode.READ)
    #     train_set, val_set = train_val_split_ui.splits.get_splits()
    # verify_train_val_sets(train_set, val_set)
    handlers_ui.stepper.set_active_step(6)
    # ----------------------------------------------------------------------------------------------- #

    # Step 6. Augmentations
    # augs = state["augmentations"]
    handlers_ui.stepper.set_active_step(7)
    # ----------------------------------------------------------------------------------------------- #

    # Step 7. Hyperparameters
    # General
    set_hyperparameters_ui(state)
    handlers_ui.stepper.set_active_step(8)

    # Step 8. Train
    train_ui.train()


def set_hyperparameters_ui(state: dict):
    # General
    n_epochs = state.get("n_epochs", 20)
    input_image_size = state.get("input_image_size", [1000, 600])
    train_batch_size = state.get("train_batch_size", 2)
    val_batch_size = state.get("val_batch_size", 1)
    val_interval = state.get("val_interval", 1)
    chart_interval = state.get("chart_interval", 1)
    # Checkpoints
    checkpoint_interval = state.get("checkpoint_interval", 1)
    keep_checkpoints = state.get("keep_checkpoints", True)
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
    use_amsgrad = state.get("use_amsgrad", False)
    beta1 = state.get("beta1", 0.9)
    beta2 = state.get("beta2", 0.999)
    sgd_momentum = state.get("sgd_momentum", 0.9)
    # LR Scheduler
    scheduler = state.get("scheduler", "Without scheduler")
    use_warmup = state.get("use_warmup", True)
    warmup_iters = state.get("warmup_iters", 1)
    warmup_ratio = state.get("warmup_ratio", 0.001)
    # MB
    model_evaluation_bm = state.get("model_evaluation_bm", True)

    # General
    hyperparameters_ui.general.epochs_input.value = n_epochs
    hyperparameters_ui.general.bigger_size_input.value = input_image_size
    hyperparameters_ui.general.smaller_size_input.value = train_batch_size
    hyperparameters_ui.general.bs_train_input.value = val_batch_size
    hyperparameters_ui.general.bs_val_input.value = val_interval
    hyperparameters_ui.general.chart_update_input.value = chart_interval
    # Checkpoints
    hyperparameters_ui.checkpoints.checkpoint_interval_input.value = checkpoint_interval
    if keep_checkpoints is True:
        hyperparameters_ui.checkpoints.checkpoint_save_switch.on()
    else:
        hyperparameters_ui.checkpoints.checkpoint_save_switch.off()
    hyperparameters_ui.checkpoints.checkpoint_save_count_input.value = max_keep_checkpoints
    if save_best_checkpoint is True:
        hyperparameters_ui.checkpoints.checkpoint_best_switch.on()
    else:
        hyperparameters_ui.checkpoints.checkpoint_best_switch.off()
    if save_last_checkpoint is True:
        hyperparameters_ui.checkpoints.checkpoint_last_switch.on()
    else:
        hyperparameters_ui.checkpoints.checkpoint_last_switch.off()
    if save_optimizer_state is True:
        hyperparameters_ui.checkpoints.checkpoint_optimizer_switch.on()
    else:
        hyperparameters_ui.checkpoints.checkpoint_optimizer_switch.off()

    # Optimizer
    if override_frozen_stages is True:
        hyperparameters_ui.optimizers.frozen_stages_switch.on()
    else:
        hyperparameters_ui.optimizers.frozen_stages_switch.off()

    hyperparameters_ui.optimizers.select_optim.set_value(optimizer)
    hyperparameters_ui.optimizers.lr.value = lr
    hyperparameters_ui.optimizers.wd.value = weight_decay

    if use_clip_grad_norm is True:
        hyperparameters_ui.optimizers.apply_clip_input.on()
    else:
        hyperparameters_ui.optimizers.apply_clip_input.off()

    hyperparameters_ui.optimizers.clip_input.value = clip_grad_norm
    hyperparameters_ui.optimizers.adam_beta1.value = beta1
    hyperparameters_ui.optimizers.adam_beta2.value = beta2
    hyperparameters_ui.optimizers.sgd_momentum.value = sgd_momentum

    if use_amsgrad is True:
        hyperparameters_ui.optimizers.amsgrad_input.on()
    else:
        hyperparameters_ui.optimizers.amsgrad_input.off()
    # LR Scheduler
    hyperparameters_ui.lr_scheduler.select_scheduler.set_value(scheduler)
    if use_warmup is True:
        hyperparameters_ui.lr_scheduler.enable_warmup_input.on()
    else:
        hyperparameters_ui.lr_scheduler.enable_warmup_input.off()
    hyperparameters_ui.lr_scheduler.warmup_iterations.value = warmup_iters
    hyperparameters_ui.lr_scheduler.warmup_ratio.value = warmup_ratio
    # MB
    if model_evaluation_bm:
        hyperparameters_ui.general.run_model_benchmark_checkbox.check()
    else:
        hyperparameters_ui.general.run_model_benchmark_checkbox.uncheck()
