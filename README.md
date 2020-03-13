# Res2Net for Pose Estimation

## Update
- [2020.3.13] Res2Net_v1b based Pose Estimation results are released now.

## Introduction
This repo uses [*Simple Baselines*](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html) as the baseline method for Pose Estimation. 

[Res2Net](https://github.com/gasvn/Res2Net) is a powerful backbone architecture that can be easily implemented into state-of-the-art models by replacing the bottleneck with Res2Net module.
More detail can be found on [ "Res2Net: A New Multi-scale Backbone Architecture"](https://arxiv.org/pdf/1904.01169.pdf)

## Performance

### Results on COCO val2017
| Arch                      |Person detector | Input size   |   AP  | Ap .5 | AP .75 | AP (M) | AP (L) |
|---------------------------|----------------|--------------|-------|-------|--------|--------|--------|
| pose_resnet_50            | prdbox         |    256x192   | 0.704 | 0.886 |  0.783 |  0.671 |  0.772 |
| pose_res2net_50           | prdbox         |    256x192   | 0.715 | 0.890 |  0.793 |  0.682 |  0.784 |
| pose_resnet_50            | GTbox          |    256x192   | 0.724 | 0.915 |  0.804 |  0.697 |  0.765 | 
| pose_res2net_50           | GTbox          |    256x192   | 0.737 | 0.925 |  0.814 |  0.708 |  0.782 |
| pose_resnet_101           | prdbox         |    256x192   | 0.714 | 0.893 |  0.793 |  0.681 |  0.781 |
| pose_res2net_101          | prdbox         |    256x192   | 0.722 | 0.894 |  0.798 |  0.689 |  0.792 |
| pose_res2net_101          | GTbox          |    256x192   | 0.744 | 0.926 |  0.826 |  0.720 |  0.785 |
| **pose_res2net_v1b_50**   | prdbox         |    256x192   | 0.722 | 0.895 |  0.797 |  0.685 |  0.794 |
| **pose_res2net_v1b_50**   | GTbox          |    256x192   | 0.743 | 0.926 |  0.816 |  0.713 |  0.792 |
| **pose_res2net_101**      | prdbox         |    256x192   | 0.730 | 0.895 |  0.803 |  0.695 |  0.800 |
| **pose_res2net_101**      | GTbox          |    256x192   | 0.753 | 0.926 |  0.825 |  0.722 |  0.801 |


### Note:
- Flip test is used.
- Person detector: prdbox refers to the Person detector that has person AP of 56.4 on COCO val2017 dataset; GTbox refers to the GT of person detection.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
6. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

7. Download pretrained models of Res2Net following the instruction from [Res2Net backbone pretrained models](https://github.com/gasvn/Res2Net). Please change the path to pretrained models **(PRETRAINED: )** in config files:  `experiments/coco/res2net/res2net50_4s_26w_256x192_d256x3_adam_lr1e-3.yaml`
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- res2net50_26w_4s-06e79181.pth
            |   |-- res2net101_26w_4s-02a759a1.pth
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet101-5d3b4d8f.pth
            |   `-- resnet152-b121ed2d.pth
            |-- pose_coco
            |   |-- (pretrained model for res2net_pose will be soon available)
            |   |-- pose_resnet_101_256x192.pth
            |   |-- pose_resnet_101_384x288.pth
            |   |-- pose_resnet_152_256x192.pth
            |   |-- pose_resnet_152_384x288.pth
            |   |-- pose_resnet_50_256x192.pth
            |   `-- pose_resnet_50_384x288.pth
            `-- pose_mpii
                |-- pose_resnet_101_256x256.pth
                |-- pose_resnet_152_256x256.pth
                `-- pose_resnet_50_256x256.pth

   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing

#### Testing on COCO val2017 dataset (pretrained model for res2net_pose will be soon available)
```
python tools/test.py \
    --cfg experiments/coco/res2net/res2net50_4s_26w_256x192_d256x3_adam_lr1e-3.yaml \
    TEST.MODEL_FILE {path to pretrained model.pth} \
    TEST.USE_GT_BBOX False
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/res2net/res2net50_4s_26w_256x192_d256x3_adam_lr1e-3.yaml
```

#### Testing on MPII dataset
```
python tools/test.py \
    --cfg experiments/mpii/res2net/res2net50_256x256_d256x3_adam_lr1e-3.yaml \
    TEST.MODEL_FILE {path to pretrained model.pth}
```

#### Training on MPII dataset

```
python tools/train.py \
    --cfg experiments/mpii/res2net/res2net50_256x256_d256x3_adam_lr1e-3.yaml
```

### Applications
Other applications such as Classification, Instance segmentation, Object detection, Semantic segmentation, Salient object detection, Class activation map can be found on https://mmcheng.net/res2net/ and https://github.com/gasvn/Res2Net .

### Citation
If you find this work or code is helpful in your research, please cite:
```
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2020},
  doi={10.1109/TPAMI.2019.2938758}, 
}
```
# Acknowledge
The code for pose estimation is partly borrowed from [Simple Baselines for Human Pose Estimation and Tracking](https://github.com/microsoft/human-pose-estimation.pytorch).
