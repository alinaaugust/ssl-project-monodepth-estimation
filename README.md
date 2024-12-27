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

## Credits

This repository is partially based on the [official implementation](https://github.com/nianticlabs/monodepth2).

#### Fulfilled by:

Maryna Horbach & Avgustyonok Alina.
