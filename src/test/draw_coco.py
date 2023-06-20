import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import supervisely as sly
import cv2


def draw_ann(coco_obj: COCO, img_id, name):
    img_ids = [img_id]
    imgs = coco_obj.loadImgs(img_ids)
    anns = coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=img_ids))
    mask = np.zeros((imgs[0]["height"], imgs[0]["width"]), dtype=bool)
    for ann in anns:
        mask = mask | mask_utils.decode(ann["segmentation"])
        rect = ann["bbox"].copy()
        rect[2] += rect[0]
        rect[3] += rect[1]
        rect = np.array(rect).astype(int).tolist()
        cv2.rectangle(
            mask,
            rect[:2],
            rect[2:],
            [
                1,
            ],
            2,
        )
        cv2.putText(
            mask,
            str(ann["category_id"]),
            rect[:2],
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            [
                1,
            ],
            2,
        )
    sly.image.write(name, mask * 255)
