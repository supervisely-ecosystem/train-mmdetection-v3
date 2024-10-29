import os

import src.sly_globals as g

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
    # Unpack state
    input_settings = state["input"]
    model_settings = state["model"]
    hyperparameters_settings = state["hyperparameters"]

    # Log settings
    sly.logger.debug(f"Input settings: {input_settings}")
    sly.logger.debug(f"Model settings: {model_settings}")
    sly.logger.debug(f"Hyperparameters settings: {hyperparameters_settings}")

    # Step 1. Input project
    set_input_ui(input_settings)
    # Step 2. Select task type
    set_task_ui(model_settings)
    # Step 3. Select arch and model
    set_model_ui(model_settings)
    # Step 4. Select classes
    set_classes_ui(classes_settings={})
    # Step 5. Split the data
    set_train_val_split_ui(train_val_split_settings={})
    # Step 6. Augmentations
    set_augmentations_ui(augmentations_settings={})
    # Step 7. Hyperparameters
    set_hyperparameters_ui(hyperparameters_settings)
    # Step 8. Train
    handlers_ui.start_train()


def set_input_ui(input_settings: dict):
    project_id = input_settings["project_id"]
    if project_id != g.project_info.id:
        raise ValueError(
            f"Project ID {project_id} in yaml config does not match the current project ID in the app session: {g.project_info.id}"
        )

    use_cache = input_settings.get("use_cache", True)
    input_project_ui.project_th.set(g.project_info)
    if use_cache is True:
        input_project_ui.use_cache_checkbox.check()
    else:
        input_project_ui.use_cache_checkbox.uncheck()


def set_task_ui(model_settings: dict):
    task_type = model_settings["task_type"].lower().capitalize()
    task_ui.task_selector.set_value(task_type)
    models_ui.update_architecture(task_ui.task_selector.get_value())
    model_leaderboard_ui.update_table(models_ui.models_meta, task_type)
    handlers_ui.select_task()


def set_model_ui(model_settings: dict):
    arch_type = model_settings["arch_type"]
    model_source = model_settings["model_source"]
    model_name = model_settings["model_name"]
    train_mode = model_settings["train_mode"]

    models_ui.arch_select.set_value(arch_type)
    models_ui.update_models(models_ui.arch_select.get_value())

    models_ui.radio_tabs.set_active_tab(model_source)
    table_data = models_ui.table.get_json_data()
    model_idx = None
    for idx, data in enumerate(table_data["raw_rows_data"]):
        if data[0] == model_name:
            model_idx = idx
            break
    if model_idx is None:
        raise ValueError(
            f"Model {model_name} not found in the table. Check if you have selected correct task type and architecture"
        )
    models_ui.table.select_row(model_idx)
    models_ui.update_selected_model(models_ui.table.get_selected_row())

    if train_mode == "finetune":
        models_ui.load_from.on()
    else:
        models_ui.load_from.off()
    handlers_ui.on_model_selected()


def set_classes_ui(classes_settings: dict):
    # selected_classes = [cls.name for cls in project_meta.obj_classes]
    # n_classes = len(classes_ui.classes.get_selected_classes())
    # if n_classes > 1:
    #     sly.logger.info(f"{n_classes} classes were selected successfully")
    # else:
    #     sly.logger.info(f"{n_classes} class was selected successfully")
    handlers_ui.select_classes()


def set_train_val_split_ui(train_val_split_settings: dict):
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
    handlers_ui.select_splits()


def set_augmentations_ui(augmentations_settings: dict):
    handlers_ui.select_augs()


def set_hyperparameters_ui(hyperparameters_settings: dict):
    # Unpack settings
    general_settings = hyperparameters_settings["general"]
    checkpoint_settings = hyperparameters_settings["checkpoint"]
    optimizer_settings = hyperparameters_settings["optimizer"]
    lr_scheduler_settings = hyperparameters_settings["lr_scheduler"]
    evaluation_settings = hyperparameters_settings["evaluation"]

    # Get settings
    # General
    n_epochs = general_settings.get("n_epochs", 20)
    input_image_size = general_settings.get("input_image_size", [1000, 600])
    train_batch_size = general_settings.get("train_batch_size", 2)
    val_batch_size = general_settings.get("val_batch_size", 1)
    val_interval = general_settings.get("val_interval", 1)
    chart_interval = general_settings.get("chart_interval", 1)

    # Checkpoints
    checkpoint_interval = checkpoint_settings.get("checkpoint_interval", 1)
    keep_checkpoints = checkpoint_settings.get("keep_checkpoints", True)
    max_keep_checkpoints = checkpoint_settings.get("max_keep_checkpoints", 3)
    save_last_checkpoint = checkpoint_settings.get("save_last_checkpoint", True)
    save_best_checkpoint = checkpoint_settings.get("save_best_checkpoint", True)
    save_optimizer_state = checkpoint_settings.get("save_optimizer_state", False)

    # Optimizer
    override_frozen_stages = optimizer_settings.get("override_frozen_stages", False)
    optimizer = optimizer_settings.get("optimizer", "AdamW")
    lr = optimizer_settings.get("lr", 0.0001)
    weight_decay = optimizer_settings.get("weight_decay", 0.0001)
    use_clip_grad_norm = optimizer_settings.get("use_clip_grad_norm", True)
    clip_grad_norm = optimizer_settings.get("clip_grad_norm", 0.1)
    # Adam
    adam_settings = optimizer_settings.get("adam", {})
    betas = adam_settings.get("betas", {"beta1": 0.9, "beta2": 0.999})
    beta1 = betas["beta1"]
    beta2 = betas["beta2"]
    use_amsgrad = adam_settings.get("use_amsgrad", False)
    # SGD
    sgd_settings = optimizer_settings.get("sgd", {})
    sgd_momentum = sgd_settings.get("sgd_momentum", 0.9)

    # LR Scheduler
    scheduler = lr_scheduler_settings.get("scheduler", "empty")
    use_warmup = lr_scheduler_settings.get("use_warmup", True)
    warmup_iters = lr_scheduler_settings.get("warmup_iters", 1)
    warmup_ratio = lr_scheduler_settings.get("warmup_ratio", 0.001)
    # StepLR
    step_lr_settings = lr_scheduler_settings.get("step_lr", {})
    step_lr_step_size = step_lr_settings.get("step_size", 1)
    step_lr_gamma = step_lr_settings.get("gamma", 0.1)
    # MultiStepLR
    multi_step_lr_settings = lr_scheduler_settings.get("multi_step_lr", {})
    multi_step_lr_milestones = multi_step_lr_settings.get("milestones", [16, 22])
    multi_step_lr_gamma = multi_step_lr_settings.get("gamma", 0.1)
    # ExponentialLR
    exponential_lr_settings = lr_scheduler_settings.get("exponential_lr", {})
    exponential_lr_gamma = exponential_lr_settings.get("gamma", 0.1)
    # ReduceLROnPlateau
    reduce_lr_on_plateau_settings = lr_scheduler_settings.get("reduce_lr_on_plateau", {})
    reduce_lr_on_plateau_factor = reduce_lr_on_plateau_settings.get("factor", 0.1)
    reduce_lr_on_plateau_patience = reduce_lr_on_plateau_settings.get("patience", 10)
    # CosineAnnealingLR
    cosine_annealing_lr_settings = lr_scheduler_settings.get("cosine_annealing_lr", {})
    cosine_annealing_lr_t_max = cosine_annealing_lr_settings.get("t_max", 1)
    cosine_annealing_lr_use_min_lr = cosine_annealing_lr_settings.get("use_min_lr", True)
    cosine_annealing_lr_min_lr = cosine_annealing_lr_settings.get("min_lr", 0)
    cosine_annealing_lr_min_lr_ratio = cosine_annealing_lr_settings.get("min_lr_ratio", 0)
    # CosineAnnealingWarmRestarts
    cosine_annealing_warm_restarts_lr_settings = lr_scheduler_settings.get(
        "cosine_annealing_warm_restarts_lr", {}
    )
    cosine_annealing_warm_restarts_lr_periods = cosine_annealing_warm_restarts_lr_settings.get(
        "periods", 1
    )
    cosine_annealing_warm_restarts_lr_restart_weights = (
        cosine_annealing_warm_restarts_lr_settings.get("restart_weights", 1)
    )
    cosine_annealing_warm_restarts_lr_use_min_lr = cosine_annealing_warm_restarts_lr_settings.get(
        "use_min_lr", True
    )
    cosine_annealing_warm_restarts_lr_min_lr = cosine_annealing_warm_restarts_lr_settings.get(
        "min_lr", 0
    )
    cosine_annealing_warm_restarts_lr_min_lr_ratio = cosine_annealing_warm_restarts_lr_settings.get(
        "min_lr_ratio", 0
    )
    # LinearLR
    linear_lr_settings = lr_scheduler_settings.get("linear_lr", {})
    linear_lr_start_factor = linear_lr_settings.get("start_factor", 0.333)
    linear_lr_end_factor = linear_lr_settings.get("end_factor", 1)
    # PolynomialLR
    polynomial_lr_settings = lr_scheduler_settings.get("polynomial_lr", {})
    polynomial_lr_min_lr = polynomial_lr_settings.get("min_lr", 0)
    polynomial_lr_power = polynomial_lr_settings.get("power", 1)
    # Model evaluation settings
    model_evaluation_bm = evaluation_settings.get("model_evaluation_bm", True)

    # Set settings
    # General
    hyperparameters_ui.general.epochs_input.value = n_epochs
    hyperparameters_ui.general.bigger_size_input.value = input_image_size[0]
    hyperparameters_ui.general.smaller_size_input.value = input_image_size[1]
    hyperparameters_ui.general.bs_train_input.value = train_batch_size
    hyperparameters_ui.general.bs_val_input.value = val_batch_size
    hyperparameters_ui.general.validation_input.value = val_interval
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
    update_optimizer_widgets(optimizer)

    hyperparameters_ui.optimizers.lr.value = lr
    hyperparameters_ui.optimizers.wd.value = weight_decay

    if use_clip_grad_norm is True:
        hyperparameters_ui.optimizers.apply_clip_input.on()
    else:
        hyperparameters_ui.optimizers.apply_clip_input.off()
    hyperparameters_ui.optimizers.clip_input.value = clip_grad_norm

    # Adam optimizer
    hyperparameters_ui.optimizers.adam_beta1.value = beta1
    hyperparameters_ui.optimizers.adam_beta2.value = beta2
    if use_amsgrad is True:
        hyperparameters_ui.optimizers.amsgrad_input.on()
    else:
        hyperparameters_ui.optimizers.amsgrad_input.off()

    # SGD optimizer
    hyperparameters_ui.optimizers.sgd_momentum.value = sgd_momentum

    # LR Scheduler
    hyperparameters_ui.lr_scheduler.select_scheduler.set_value(scheduler)
    update_scheduler_widgets(scheduler)
    if use_warmup is True:
        hyperparameters_ui.lr_scheduler.enable_warmup_input.on()
    else:
        hyperparameters_ui.lr_scheduler.enable_warmup_input.off()
    hyperparameters_ui.lr_scheduler.warmup_iterations.value = warmup_iters
    hyperparameters_ui.lr_scheduler.warmup_ratio.value = warmup_ratio

    # StepLR
    hyperparameters_ui.lr_scheduler.step_input.value = step_lr_step_size
    hyperparameters_ui.lr_scheduler.step_gamma_input.value = step_lr_gamma
    # MultiStepLR
    hyperparameters_ui.lr_scheduler.multi_steps_input.set_value(multi_step_lr_milestones)
    hyperparameters_ui.lr_scheduler.multi_steps_gamma_input.value = multi_step_lr_gamma
    # ExponentialLR
    hyperparameters_ui.lr_scheduler.exp_gamma_input.value = exponential_lr_gamma
    # ReduceLROnPlateau
    hyperparameters_ui.lr_scheduler.reduce_plateau_factor_input.value = reduce_lr_on_plateau_factor
    hyperparameters_ui.lr_scheduler.reduce_plateau_patience_input.value = (
        reduce_lr_on_plateau_patience
    )
    # CosineAnnealingLR
    hyperparameters_ui.lr_scheduler.cosineannealing_tmax_input.value = cosine_annealing_lr_t_max
    if cosine_annealing_lr_use_min_lr is True:
        hyperparameters_ui.lr_scheduler.etamin_switch_input.on()
    else:
        hyperparameters_ui.lr_scheduler.etamin_switch_input.off()
    hyperparameters_ui.lr_scheduler.etamin_input.value = cosine_annealing_lr_min_lr
    hyperparameters_ui.lr_scheduler.etamin_ratio_input.value = cosine_annealing_lr_min_lr_ratio
    # CosineAnnealingWarmRestarts
    hyperparameters_ui.lr_scheduler.cosinerestart_periods_input.set_value(
        cosine_annealing_warm_restarts_lr_periods
    )
    hyperparameters_ui.lr_scheduler.cosinerestart_restart_weights_input.set_value(
        cosine_annealing_warm_restarts_lr_restart_weights
    )
    if cosine_annealing_warm_restarts_lr_use_min_lr is True:
        hyperparameters_ui.lr_scheduler.etamin_switch_input.on()
    else:
        hyperparameters_ui.lr_scheduler.etamin_switch_input.off()
    hyperparameters_ui.lr_scheduler.etamin_input.value = cosine_annealing_warm_restarts_lr_min_lr
    hyperparameters_ui.lr_scheduler.etamin_ratio_input.value = (
        cosine_annealing_warm_restarts_lr_min_lr_ratio
    )
    # LinearLR
    hyperparameters_ui.lr_scheduler.linear_start_factor_input.value = linear_lr_start_factor
    hyperparameters_ui.lr_scheduler.linear_end_factor_input.value = linear_lr_end_factor
    # PolynomialLR
    hyperparameters_ui.lr_scheduler.poly_eta_input.value = polynomial_lr_min_lr
    hyperparameters_ui.lr_scheduler.poly_power_input.value = polynomial_lr_power
    # Model Benchmark
    if model_evaluation_bm:
        hyperparameters_ui.general.run_model_benchmark_checkbox.check()
    else:
        hyperparameters_ui.general.run_model_benchmark_checkbox.uncheck()
    handlers_ui.select_hyperparameters()


def update_optimizer_widgets(optimizer):
    for optim in hyperparameters_ui.optimizers.optimizers_params.keys():
        if optimizer == optim:
            hyperparameters_ui.optimizers.optimizers_params[optim].show()
        else:
            hyperparameters_ui.optimizers.optimizers_params[optim].hide()


def update_scheduler_widgets(scheduler):
    for sched in hyperparameters_ui.lr_scheduler.schedulers_params.keys():
        if scheduler == sched:
            hyperparameters_ui.lr_scheduler.schedulers_params[sched].show()
        else:
            hyperparameters_ui.lr_scheduler.schedulers_params[sched].hide()
