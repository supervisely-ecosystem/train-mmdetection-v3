<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/train-mmdetection-2/assets/115161827/a2a022a9-b1b1-4231-9a8d-37e4d3898acf"/>  

# Train MMDetection 3.0

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Obtain-saved-checkpoints">Obtain saved checkpoints</a> •
  <a href="#How-To-Use-Custom-Model-Outside-The-Platform">How To Use Custom Model Outside The Platform</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/train-mmdetection-v3)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/train-mmdetection-v3)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/train-mmdetection-v3.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/train-mmdetection-v3.png)](https://supervise.ly)

</div>

# Overview

Train MMDetection 3.0 models in Supervisely.

Application key points:
- The app supports Object Detection and Instance Segmentation tasks
- There are almost all Object Detection and Instance Segmentation models from MMDetection 3.0
- You can compare all the models performance and metrics in the Model Leaderboard table
- Fine-tune pretrained models or train it from scratch
- Define Train / Validation splits
- Select classes for training
- Define augmentations
- Tune hyperparameters
- Preview LR schedulers before start the training
- Watch the training progress, losses, metrics in charts
- Save training checkpoints to Team Files

**The app only supports the models in Object Detection and Instance Segmentation tasks [(origianl model zoo)](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/model_zoo.md):**

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Instance Segmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/fast_rcnn">Fast R-CNN (ICCV'2015)</a></li>
            <li><a href="configs/faster_rcnn">Faster R-CNN (NeurIPS'2015)</a></li>
            <li><a href="configs/rpn">RPN (NeurIPS'2015)</a></li>
            <li><a href="configs/ssd">SSD (ECCV'2016)</a></li>
            <li><a href="configs/retinanet">RetinaNet (ICCV'2017)</a></li>
            <li><a href="configs/cascade_rcnn">Cascade R-CNN (CVPR'2018)</a></li>
            <li><a href="configs/yolo">YOLOv3 (ArXiv'2018)</a></li>
            <li><a href="configs/cornernet">CornerNet (ECCV'2018)</a></li>
            <li><a href="configs/grid_rcnn">Grid R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/guided_anchoring">Guided Anchoring (CVPR'2019)</a></li>
            <li><a href="configs/fsaf">FSAF (CVPR'2019)</a></li>
            <li><a href="configs/centernet">CenterNet (CVPR'2019)</a></li>
            <li><a href="configs/libra_rcnn">Libra R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/tridentnet">TridentNet (ICCV'2019)</a></li>
            <li><a href="configs/fcos">FCOS (ICCV'2019)</a></li>
            <li><a href="configs/reppoints">RepPoints (ICCV'2019)</a></li>
            <li><a href="configs/free_anchor">FreeAnchor (NeurIPS'2019)</a></li>
            <li><a href="configs/cascade_rpn">CascadeRPN (NeurIPS'2019)</a></li>
            <li><a href="configs/foveabox">Foveabox (TIP'2020)</a></li>
            <li><a href="configs/double_heads">Double-Head R-CNN (CVPR'2020)</a></li>
            <li><a href="configs/atss">ATSS (CVPR'2020)</a></li>
            <li><a href="configs/nas_fcos">NAS-FCOS (CVPR'2020)</a></li>
            <li><a href="configs/centripetalnet">CentripetalNet (CVPR'2020)</a></li>
            <li><a href="configs/autoassign">AutoAssign (ArXiv'2020)</a></li>
            <li><a href="configs/sabl">Side-Aware Boundary Localization (ECCV'2020)</a></li>
            <li><a href="configs/dynamic_rcnn">Dynamic R-CNN (ECCV'2020)</a></li>
            <li><a href="configs/detr">DETR (ECCV'2020)</a></li>
            <li><a href="configs/paa">PAA (ECCV'2020)</a></li>
            <li><a href="configs/vfnet">VarifocalNet (CVPR'2021)</a></li>
            <li><a href="configs/sparse_rcnn">Sparse R-CNN (CVPR'2021)</a></li>
            <li><a href="configs/yolof">YOLOF (CVPR'2021)</a></li>
            <li><a href="configs/yolox">YOLOX (CVPR'2021)</a></li>
            <li><a href="configs/deformable_detr">Deformable DETR (ICLR'2021)</a></li>
            <li><a href="configs/tood">TOOD (ICCV'2021)</a></li>
            <li><a href="configs/ddod">DDOD (ACM MM'2021)</a></li>
            <li><a href="configs/rtmdet">RTMDet (ArXiv'2022)</a></li>
            <li><a href="configs/conditional_detr">Conditional DETR (ICCV'2021)</a></li>
            <li><a href="configs/dab_detr">DAB-DETR (ICLR'2022)</a></li>
            <li><a href="configs/dino">DINO (ICLR'2023)</a></li>
      </ul>
      <ul>
          <b>Distillation</b>
          <li><a href="configs/ld">Localization Distillation (CVPR'2022)</a></li>
          <li><a href="configs/lad">Label Assignment Distillation (WACV'2022)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/mask_rcnn">Mask R-CNN (ICCV'2017)</a></li>
          <li><a href="configs/cascade_rcnn">Cascade Mask R-CNN (CVPR'2018)</a></li>
          <li><a href="configs/ms_rcnn">Mask Scoring R-CNN (CVPR'2019)</a></li>
          <li><a href="configs/htc">Hybrid Task Cascade (CVPR'2019)</a></li>
          <li><a href="configs/yolact">YOLACT (ICCV'2019)</a></li>
          <li><a href="configs/instaboost">InstaBoost (ICCV'2019)</a></li>
          <li><a href="configs/solo">SOLO (ECCV'2020)</a></li>
          <li><a href="configs/point_rend">PointRend (CVPR'2020)</a></li>
          <li><a href="configs/detectors">DetectoRS (ArXiv'2020)</a></li>
          <li><a href="configs/solov2">SOLOv2 (NeurIPS'2020)</a></li>
          <li><a href="configs/scnet">SCNet (AAAI'2021)</a></li>
          <li><a href="configs/queryinst">QueryInst (ICCV'2021)</a></li>
          <li><a href="configs/mask2former">Mask2Former (ArXiv'2021)</a></li>
          <li><a href="configs/condinst">CondInst (ECCV'2020)</a></li>
          <li><a href="projects/SparseInst">SparseInst (CVPR'2022)</a></li>
          <li><a href="configs/rtmdet">RTMDet (ArXiv'2022)</a></li>
          <li><a href="configs/boxinst">BoxInst (CVPR'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>VGG (ICLR'2015)</li>
        <li>ResNet (CVPR'2016)</li>
        <li>ResNeXt (CVPR'2017)</li>
        <li>MobileNetV2 (CVPR'2018)</li>
        <li><a href="configs/hrnet">HRNet (CVPR'2019)</a></li>
        <li><a href="configs/empirical_attention">Generalized Attention (ICCV'2019)</a></li>
        <li><a href="configs/gcnet">GCNet (ICCVW'2019)</a></li>
        <li><a href="configs/res2net">Res2Net (TPAMI'2020)</a></li>
        <li><a href="configs/regnet">RegNet (CVPR'2020)</a></li>
        <li><a href="configs/resnest">ResNeSt (ArXiv'2020)</a></li>
        <li><a href="configs/pvt">PVT (ICCV'2021)</a></li>
        <li><a href="configs/swin">Swin (CVPR'2021)</a></li>
        <li><a href="configs/pvt">PVTv2 (ArXiv'2021)</a></li>
        <li><a href="configs/resnet_strikes_back">ResNet strikes back (ArXiv'2021)</a></li>
        <li><a href="configs/efficientnet">EfficientNet (ArXiv'2021)</a></li>
        <li><a href="configs/convnext">ConvNeXt (CVPR'2022)</a></li>
        <li><a href="projects/ConvNeXt-V2">ConvNeXtv2 (ArXiv'2023)</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/pafpn">PAFPN (CVPR'2018)</a></li>
        <li><a href="configs/nas_fpn">NAS-FPN (CVPR'2019)</a></li>
        <li><a href="configs/carafe">CARAFE (ICCV'2019)</a></li>
        <li><a href="configs/fpg">FPG (ArXiv'2020)</a></li>
        <li><a href="configs/groie">GRoIE (ICPR'2020)</a></li>
        <li><a href="configs/dyhead">DyHead (CVPR'2021)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/ghm">GHM (AAAI'2019)</a></li>
          <li><a href="configs/gfl">Generalized Focal Loss (NeurIPS'2020)</a></li>
          <li><a href="configs/seesaw_loss">Seasaw Loss (CVPR'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/faster_rcnn/faster-rcnn_r50_fpn_ohem_1x_coco.py">OHEM (CVPR'2016)</a></li>
          <li><a href="configs/gn">Group Normalization (ECCV'2018)</a></li>
          <li><a href="configs/dcn">DCN (ICCV'2017)</a></li>
          <li><a href="configs/dcnv2">DCNv2 (CVPR'2019)</a></li>
          <li><a href="configs/gn+ws">Weight Standardization (ArXiv'2019)</a></li>
          <li><a href="configs/pisa">Prime Sample Attention (CVPR'2020)</a></li>
          <li><a href="configs/strong_baselines">Strong Baselines (CVPR'2021)</a></li>
          <li><a href="configs/resnet_strikes_back">Resnet strikes back (ArXiv'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

# How to Run

**Step 1.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 2.** Select the MMDetection task you need to solve 

<img src="https://github.com/supervisely-ecosystem/train-mmdetection-v3/assets/115161827/136a8a5e-4066-4a1f-86f8-b14e266527b7" width="100%" style='padding-top: 10px'>  

**Step 3.** Choose the pretrained or custom object detection model

<img src="https://github.com/supervisely-ecosystem/train-mmdetection-v3/assets/115161827/b07114db-d620-469b-893f-202d3ce356c6" width="100%" style='padding-top: 10px'>  

**Step 4.** Select the classes you want to train MMDetection on

<img src="https://github.com/supervisely-ecosystem/train-mmdetection-v3/assets/115161827/29b0b4ab-44a5-4d1f-92f6-d5f0aee54b77" width="100%" style='padding-top: 10px'>  

**Step 5.** Define the train/val splits

<img src="https://github.com/supervisely-ecosystem/train-mmdetection-v3/assets/115161827/ea58fe7d-c592-43b0-8492-8535117d5a06" width="100%" style='padding-top: 10px'>  

**Step 6.** Choose either ready-to-use augmentation template or provide custom pipeline

<img src="https://github.com/supervisely-ecosystem/train-mmdetection-v3/assets/115161827/18664fa4-398c-4848-b252-9f22b93055d5" width="100%" style='padding-top: 10px'>  

**Step 7.** Configure the training parameters

<img src="https://github.com/supervisely-ecosystem/train-mmdetection-v3/assets/115161827/a7c9e642-0488-4175-967a-e1d1f2727efb" width="100%" style='padding-top: 10px'>  

**Step 8.** Click `Train` button and observe the training progress, metrics charts and visualizations 

<img src="https://github.com/supervisely-ecosystem/train-mmdetection-v3/assets/115161827/6354d252-a1ee-4046-9d66-1881ad64c17f" width="100%" style='padding-top: 10px'>  

# Obtain saved checkpoints

All the trained checkpoints, that are generated through the process of training models are stored in [Team Files](https://app.supervise.ly/files/) in the folder **mmdetection-3**.

To navigate to team files, go to the `Start` menu and press the `Team files` button

![screenshot-dev-supervise-ly-workspaces-1689091116341 copy](https://github.com/supervisely-ecosystem/train-mmdetection-v3/assets/115161827/f57ac1ea-8d05-4a35-a924-5c9e75d9617f)


# How To Use Custom Model Outside The Platform

We have a [Jupyter Notebook](https://github.com/supervisely-ecosystem/serve-mmdetection-v3/blob/master/inference_outside_sly.ipynb) as an example of how you can use your custom trained model outside Supervisely.
First, download the `config.py` file and model weights (`.pth`) from Superviesly Team Files. Then, run the [notebook](https://github.com/supervisely-ecosystem/serve-mmdetection-v3/blob/master/inference_outside_sly.ipynb) and follow its instructions.

A base code example is here:
```python
# Put your paths here:
img_path = "demo_data/image_01.jpg"
config_path = "app_data/work_dir/config.py"
weights_path = "app_data/work_dir/epoch_8.pth"

device = "cuda:0"

import mmcv
from mmengine import Config
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.visualization.local_visualizer import DetLocalVisualizer
from PIL import Image

# build the model
cfg = Config.fromfile(config_path)
model = init_detector(cfg, weights_path, device=device, palette='random')

# predict
result = inference_detector(model, img_path)
print(result)

# visualize
img = mmcv.imread(img_path, channel_order="rgb")
visualizer: DetLocalVisualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
visualizer.add_datasample("result", img, data_sample=result, draw_gt=False, wait_time=0, show=False)
res_img = visualizer.get_image()
Image.fromarray(res_img)
```


# Acknowledgment

This app is based on the great work `MMDetection` ([github](https://github.com/open-mmlab/mmdetection)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmdetection?style=social)
