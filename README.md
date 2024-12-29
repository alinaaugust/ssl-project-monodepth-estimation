# ssl-project-monodepth-estimation

This repository contains our attempt of implementing the [Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/pdf/1806.01260v4). Corresponding files contain full training pipeline; in [this notebook](/depth_estimation_example.ipynb) you can find inference example for the pretrained models from [here](https://github.com/nianticlabs/monodepth2).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Get the KITTI dataset
  - if you want to use all training data (be careful, the dataset weights >170GB):
   ```bash
   wget -i splits/kitti_archives_to_download.txt -P kitti_data/
   ```
  - if you want to run our example notebook with the authors' pretrained models (this line is also duplicated in the notebook, so this step is optional):
   ```bash
   wget -i splits/kitti_example_archives.txt -P kitti_data/
   ```
  - pre-processing
   ```bash
   cd kitti_data
   unzip "*.zip"
   cd ..
   find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
   ```

## System Description

The primary goal of the project is to replicate the results of the method described in the paper by Godard et al. (2019). The approach involves the use of three interconnected models, each with specific architectural features and purposes.

## Training

To start training with default parameters, run the following command:

   python train.py  

This will initialize the training process using the predefined configuration. Make sure all necessary files and dependencies are properly set up before running the script.

The general workflow of the system is illustrated below:

### Encoder

The encoder used for all models is based on ResNet18, initialized with weights pretrained on ImageNet. While ResNet50 has been more commonly used in prior works, the authors opted for a lighter architecture to balance performance and computational efficiency.

Since the Pose Network (discussed later) takes two images as input, the encoder uses a modified structure where weights are duplicated to handle the dual inputs. The decoder, however, offers flexibility: it can either be shared across all models or trained separately for each.

### Depth Network

The Depth Network is responsible for predicting disparity, which is later converted to depth. This network is implemented as a fully-convolutional UNet architecture. It takes a single image as input and outputs the disparity map.

### Pose Network

The Pose Network predicts the relative motion (pose change) between two consecutive frames. This information is later used to train both the Pose and Depth Networks.

#### File Paths

    ResNet18 Encoder with Pretrained Weights: src/model/resnet_encoder.py
    Pose Network: src/model/pose_decoder.py
    Depth Network: src/model/depth_decoder.py
    
### Data and Training Process

The data consists of sequential frames extracted from videos, and the training process is based on pairs of consecutive frames.

1. Depth Prediction: The depth of one frame is predicted using the Depth Network.

2. Relative Motion Estimation: The Pose Network estimates the relative motion (pose change) between the two frames.

3. Depth Projection and Reconstruction: Using the estimated depth and relative motion, the depth map of one frame is projected onto the second frame. The second frame is then reconstructed based on this projection.

4. Loss Calculation: The reconstructed frame is compared with the actual frame to compute the loss functions. The weights of the Depth and Pose Networks are updated based on these loss values.

## Credits

This repository is partially based on the [official implementation](https://github.com/nianticlabs/monodepth2).

#### Fulfilled by:

Maryna Horbach & Avgustyonok Alina.
