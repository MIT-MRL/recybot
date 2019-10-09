# A Pytorch Implementation of Phone Recycling Neural Network

### Introduction

This repository is for the IROS submission paper, "See the E-Waste! Training Visual Intelligence to See Dense Circuit Boards for Recycling"

### Disclaimer

- In this repository, we provide the implementation of our vision neural network based on Pytorch. Note that our code is heavily based on [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Thanks [Roy](https://github.com/roytseng-tw) for his great work!
- We trained with image batch size 8 using one P6000

### Installation

For environment requirements, data preparation and compilation, please refer to [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).

### Usage

For training and testing, we keep the same as the one in [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). To train and test our network, simply use corresponding config files. For example, to train on the phone dataset:

```shell
python tools/train_net_step.py --dataset phone --cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_1x_edge.yaml
```

To evaluate the model, simply use:

```shell
python tools/test_net.py --dataset phone --cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_1x_edge.yaml --load_ckpt {path/to/your/checkpoint}
```
