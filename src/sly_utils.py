import src.sly_globals as g
import supervisely as sly


def download_custom_config(remote_weights_path: str):
    # download config_xxx.py
    save_dir = remote_weights_path.split("checkpoints")
    files = g.api.file.listdir(g.TEAM_ID, save_dir)
    # find config by name in save_dir
    remote_config_path = [f for f in files if f.endswith(".py")]
    assert len(remote_config_path) > 0, f"Can't find config in {save_dir}."
    remote_config_path = remote_config_path[0]
    config_name = remote_config_path.split("/")[-1]
    config_path = sly.app.get_data_dir() + f"/{config_name}"
    g.api.file.download(g.TEAM_ID, remote_config_path, config_path)
    return config_path


def download_custom_model_weights(remote_weights_path: str):
    # download .pth
    model_name = remote_weights_path.split("/")[-1]
    weights_path = sly.app.get_data_dir() + f"/{model_name}"
    g.api.file.download(g.TEAM_ID, remote_weights_path, weights_path)
    return weights_path


def download_custom_model(remote_weights_path: str):
    # save_dir structure:
    # - checkpoints
    # - logs
    # - config_xxx.py
    config_path = download_custom_config(remote_weights_path)
    weights_path = download_custom_model_weights(remote_weights_path)
    return weights_path, config_path


def upload_artifacts(work_dir: str, experiment_name: str = None):
    task_id = g.api.task_id or ""
    g.api.file.upload_directory(g.TEAM_ID, work_dir, f"/mmdetction-2/{experiment_name}_{task_id}")


def download_project():
    project_dir = f"{sly.app.get_data_dir()}/sly_project"
    if sly.fs.dir_exists(project_dir):
        sly.fs.remove_dir(project_dir)
    sly.Project.download(g.api, g.PROJECT_ID, project_dir)
    return project_dir
