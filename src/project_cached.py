import asyncio
import os

import supervisely as sly
from supervisely.app.widgets import Progress
from supervisely.project.download import (
    copy_from_cache,
    download_to_cache,
    get_cache_size,
)


def _no_cache_download(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    project_dir: str,
    progress: Progress,
    semaphore: asyncio.Semaphore = None,
):
    total = project_info.items_count
    try:
        with progress(message="Downloading input data...", total=total) as pbar:
            sly.download_async(
                api=api,
                project_id=project_info.id,
                dest_dir=project_dir,
                semaphore=semaphore,
                progress_cb=pbar.update,
                save_images=True,
                save_image_info=True,
            )
    except Exception:
        api.logger.warning(
            "Failed to download project using async download. Trying sync download..."
        )
        with progress(message="Downloading input data...", total=total) as pbar:
            sly.download_project(
                api=api,
                project_id=project_info.id,
                dest_dir=project_dir,
                log_progress=True,
                progress_cb=pbar.update,
                save_images=True,
                save_image_info=True,
            )


def download_project(
    api: sly.Api,
    project_id: int,
    project_dir: str,
    use_cache: bool,
    progress: Progress,
):
    if os.path.exists(project_dir):
        sly.fs.clean_dir(project_dir)
    project_info = api.project.get_info_by_id(project_id)
    if not use_cache:
        _no_cache_download(api, project_info, project_dir, progress)
        return
    try:
        # get datasets to download and cached
        dataset_infos = api.dataset.get_list(project_id)
        # get images count
        total = sum([ds_info.images_count for ds_info in dataset_infos])
        # download
        with progress(message="Downloading input data...", total=total) as pbar:
            download_to_cache(api, project_id, progress_cb=pbar.update)

        # copy datasets from cache
        total = get_cache_size(project_info.id)
        with progress(
            message="Retreiving data from cache...",
            total=total,
            unit="B",
            unit_scale=True,
        ) as pbar:
            copy_from_cache(
                project_id=project_info.id,
                dest_dir=project_dir,
                progress_cb=pbar.update,
            )
    except Exception:
        sly.logger.warning(
            f"Failed to download project using cache. Falling back to default...", exc_info=True
        )
        if os.path.exists(project_dir):
            sly.fs.clean_dir(project_dir)
        _no_cache_download(api, project_info, project_dir, progress)
