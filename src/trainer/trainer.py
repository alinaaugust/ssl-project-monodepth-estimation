from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import (
    ResNetEncoder,
    MultiImageResNet,
    PoseEstimationDecoder,
    DepthDecoder2,
)

import json

from utils import *
from kitti_utils import *

# from layers import *

from src.datasets import KITTIDepthDataset, KITTIRAWDataset, KITTIOdomDataset
from IPython import embed


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = self.config.device

        self.models = {}
        self.model_parameters = []
        self.num_scaling_factors = len(self.config.scaling_factors)
        self.num_frames = len(self.config.frame_ids)
        self.num_pose_frames = (
            self.num_frames if self.config.pose_model_input != "pairs" else 2
        )
        self.use_pose_model = (
            False if (self.config.use_stereo and self.config.frame_ids == [0]) else True
        )

        self.logs_dir = os.path.join(self.config.logs_dir, self.config.run_name)

        if self.config.use_stereo:
            self.config.frame_ids.append("s")

        self.models["encoder"] = ResNetEncoder(
            self.config.num_layers, self.config.weights == "pretrained"
        ).to(self.device)
        self.models["depth_decoder"] = DepthDecoder2(
            self.models["encoder"].num_ch_enc, self.config.scaling_factors
        ).to(self.device)
        self.model_parameters += list(self.models["encoder"].parameters())
        self.model_parameters += list(self.models["depth_decoder"].parameters())

        if self.use_pose_model:
            if self.config.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = ResNetEncoder(
                    self.config.num_layers,
                    self.config.weigths == "pretrained",
                    num_input_images=self.num_pose_frames,
                ).to(self.device)

                self.model_parameters += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = PoseEstimationDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2,
                )

            elif self.config.pose_model_type == "shared":
                self.models["pose"] = PoseEstimationDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames
                )

        if self.config.use_predictive_mask:
            self.models["predictive_mask"] = DepthDecoder2(
                self.models["encoder"].num_ch_enc,
                self.config.scaling_factors,
                num_output_channels=(len(self.config.frame_ids) - 1),
            ).to(self.device)
            self.model_parameters += list(self.models["predictive_mask"].parameters())

        self.optimizer = optim.Adam(self.parameters_to_train, self.config.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, self.config.scheduler_step_size, 0.1
        )

        if self.config.weights_dir is not None:
            print(f"Using pretrained model weights from: {self.config.weights_dir}\n")
            self._load_model()

        print(f"Starting run: {self.config.run_name}\n")
        print("Training model: ", self.config.model_name, "\n")
        print("Models are saved in:  ", self.opt.log_dir, "\n")
        print("Current device:  ", self.device, "\n")

        self.dataset = (
            KITTIRAWDataset if self.config.dataset == "kitti" else KITTIOdomDataset
        )
        fpath = os.path.join(
            os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt"
        )

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_format = ".png" if self.config.use_png else ".jpg"

        num_train_samples = len(train_filenames)
        self.num_total_steps = (
            num_train_samples // self.config.batch_size * self.config.num_epochs
        )

        train_dataset = self.dataset(
            self.config.data_dir,
            train_filenames,
            self.config.height,
            self.config.width,
            self.config.frame_ids,
            [0, 1, 2, 3],
            split="train",
            img_format=img_format,
        )
        self.train_loader = DataLoader(
            train_dataset,
            self.config.batch_size,
            True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_dataset = self.dataset(
            self.config.data_dir,
            val_filenames,
            self.config.height,
            self.config.width,
            self.config.frame_ids,
            [0, 1, 2, 3],
            split="val",
            img_format=img_format,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.config.batch_size,
            True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_iter = iter(self.val_loader)

        # what do we have here?
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2**scale)
            w = self.opt.width // (2**scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel",
            "de/sq_rel",
            "de/rms",
            "de/log_rms",
            "da/a1",
            "da/a2",
            "da/a3",
        ]

        print("Dataset part:\n  ", self.config.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)
            )
        )

    def train(self):
        """
        Here we go again.
        """
        self.depth = 0
        self.epoch = 0
        for _ in range(self.config.num_epochs):
            self.epoch += 1
            self.train_epoch()
            if (self.epoch + 1) % self.config.save_step == 0:
                self._save_model()

    def train_epoch(self):
        """
        Single epoch with training and validation.
        """
        self.lr_scheduler.step()

        for model in self.models.values():
            model.train()

        # pum pum pum
        # for batch_idx, batch in enumerate(self.train_loader):

    def _save_model(self):
        """
        Save model state.
        """
        save_folder = os.path.join(
            self.log_path, "models", "weights_{}".format(self.epoch)
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == "encoder":
                to_save["height"] = self.config.height
                to_save["width"] = self.config.width
                to_save["use_stereo"] = self.config.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("optimizer"))
        torch.save(self.optimizer.state_dict(), save_path)

    def _load_model(self):
        """
        Load model(s) from pretrained.
        """
        self.config.weights_dir = os.path.expanduser(self.config.weights_dir)
        print("Loading model from pretrained {}".format(self.config.weights_dir))

        for n in self.config.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.config.weights_dir, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        optimizer_load_path = os.path.join(self.config.weights_dir, "optimizer.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading optimizer weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
