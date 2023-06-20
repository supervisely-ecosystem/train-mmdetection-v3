import supervisely as sly
from supervisely.app import widgets
from src.sly_dataset import SuperviselyDatasetSplit
from mmdet.datasets.transforms import LoadAnnotations
from mmcv.transforms.loading import LoadImageFromFile
from mmengine.visualization import Visualizer


api = sly.Api()

dst_dir = "sly_project2"
# sly.Project.download(api, 23275, dst_dir)
# api.file.download_directory(440, "/mmdetection/34750_test-lemons/", "model")

project = sly.Project(dst_dir, sly.OpenMode.READ)
splits = widgets.TrainValSplits(project_fs=project)
cls_table = widgets.ClassesTable(project.meta)
btn = widgets.Button()

c = widgets.Container([splits, cls_table, btn])
app = sly.Application(c)


@btn.click
def on_click():
    train_split, val_split = splits.get_splits()
    sly.json.dump_json_file(train_split, "train_split.json")
    sly.json.dump_json_file(val_split, "val_split.json")
    selected_classes = cls_table.get_selected_classes()

    task = "instance_segmentation"
    pipeline = [LoadImageFromFile(), LoadAnnotations(with_mask=task == "instance_segmentation")]

    ds = SuperviselyDatasetSplit(
        dst_dir,
        "train_split.json",
        task,
        selected_classes=selected_classes,
        pipeline=pipeline,
    )
    sample = ds[0]

    vis = Visualizer(save_dir="vis")
    vis.set_image(sample["img"])
    vis.draw_bboxes(sample["gt_bboxes"].tensor)
    if task == "instance_segmentation":
        vis.draw_binary_masks(sample["gt_masks"].masks.astype(bool))
    sly.image.write("test.png", vis.get_image()[..., ::-1])
