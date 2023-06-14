from mmengine.dataset import BaseDataset, Compose
from mmdet.registry import DATASETS
from typing import List
import numpy as np
import supervisely as sly
from supervisely.project.project import ItemInfo
import pycocotools.mask


@DATASETS.register_module()
class SuperviselyDatasetSplit(BaseDataset):
    def __init__(
        self,
        data_root: str,
        split_file: str,
        selected_classes: list = None,
        filter_images_without_gt: bool = False,
        save_coco_ann_file: str = None,
        serialize_data: bool = True,
        test_mode: bool = False,
        pipeline: list = [],
        max_refetch=1000,
    ):
        self.data_root = data_root
        self.split_file = split_file
        self.selected_classes = selected_classes
        self.filter_images_without_gt = filter_images_without_gt
        self.save_coco_ann_file = save_coco_ann_file
        self.item_infos = sly.json.load_json_file(self.split_file)
        self.project = sly.Project(self.data_root, sly.OpenMode.READ)
        self._metainfo = self._get_metainfo_sly(self.selected_classes)
        self.class2id = {c: i for i, c in enumerate(self._metainfo["classes"])}

        # from base class init:
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # TODO: implement filtering (when not all classes are selected)
        if self.filter_images_without_gt is True:
            raise NotImplementedError()

        # Build pipeline.
        self.pipeline = Compose(pipeline)

        # Full initialize the dataset.
        self.full_init()

    def full_init(self):
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Save coco ann_file if needed
        if isinstance(self.save_coco_ann_file, str):
            coco_json = self._create_coco_json()
            sly.json.dump_json_file(coco_json, self.save_coco_ann_file, indent=2)
        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()
        self._fully_initialized = True

    def _get_metainfo_sly(self, selected_classes: list = None):
        obj_classes = self.project.meta.obj_classes
        if selected_classes is None:
            selected_classes = list(obj_classes.keys())
        classes = [obj_classes.get(cls).name for cls in selected_classes]
        palette = [tuple(obj_classes.get(cls).color) for cls in selected_classes]
        metainfo = {"classes": classes, "palette": palette}
        return metainfo

    def load_data_list(self) -> List[dict]:
        data_list = []
        for i, item_info in enumerate(self.item_infos):
            item_info = ItemInfo(*item_info)
            ann = sly.Annotation.load_json_file(item_info.ann_path, self.project.meta)

            # TODO: implement filtering
            # if self.selected_classes is not None and self.filter_images_without_gt:
            #     ann = ann.filter_labels_by_classes(keep_classes=self.selected_classes)

            instances = []
            for label in ann.labels:
                if not isinstance(label.geometry, sly.Bitmap):
                    continue
                rect: sly.Rectangle = label.geometry.to_bbox()
                bbox = [rect.left, rect.top, rect.right, rect.bottom]
                bbox_cls = self.class2id[label.obj_class.name]
                mask = np.zeros(ann.img_size, bool)
                bitmap = label.geometry
                bitmap.draw(mask, 1, 0)
                rle = pycocotools.mask.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode()
                instance = {"mask": rle, "bbox": bbox, "bbox_label": bbox_cls, "ignore_flag": 0}
                instances.append(instance)
            data_item = {
                "img_id": i + 1,
                "img_path": item_info.img_path,
                "instances": instances,
                "height": ann.img_size[0],
                "width": ann.img_size[1],
            }
            data_list.append(data_item)
        return data_list

    def _create_coco_json(self):
        image_infos = []
        annotations = []

        instance_id = 0
        for i, data_item in enumerate(self.data_list):
            image_id = i + 1

            image_info = {
                "id": image_id,
                "height": data_item["height"],
                "width": data_item["width"],
            }
            image_infos.append(image_info)

            for instance in data_item["instances"]:
                instance_id += 1
                bbox = instance["bbox"].copy()
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                annotation = {
                    "id": instance_id,  # unique id for each annotation
                    "image_id": image_id,
                    "category_id": instance["bbox_label"],
                    "segmentation": instance["mask"],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
                annotations.append(annotation)

        # Create COCO format dictionary
        categories = self._get_coco_categories()
        coco_json = {
            "info": {},
            "licenses": [],
            "categories": categories,
            "images": image_infos,
            "annotations": annotations,
        }
        return coco_json

    def _get_coco_categories(self):
        # won't match to the dataset's class ids
        return [{"id": id, "name": class_name} for class_name, id in self.class2id.items()]
