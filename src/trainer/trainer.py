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
    DepthDecoder,
)

import json

from main_utils import *
from kitti_utils import *

# from layers import *

from src.datasets import KITTIDepthDataset, KITTIRAWDataset, KITTIOdomDataset
from src.loss import SSIMLoss, calculate_losses, depth_errors
from src.model import BackprojectDepth, Project3D
from IPython import embed


class Trainer:
    def __init__(self, writer, config):
        self.config = config
        self.writer = writer
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
        self.models["depth_decoder"] = DepthDecoder(
            self.models["encoder"].num_ch_enc, self.config.scaling_factors
        ).to(self.device)
        self.model_parameters += list(self.models["encoder"].parameters())
        self.model_parameters += list(self.models["depth_decoder"].parameters())

        if self.use_pose_model:
            if self.config.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = ResNetEncoder(
                    self.config.num_layers,
                    self.config.weights == "pretrained",
                    num_input_images=self.num_pose_frames,
                ).to(self.device)

                self.model_parameters += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = PoseEstimationDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    input_feature_count=1,
                    frames_to_predict=2,
                )

            elif self.config.pose_model_type == "shared":
                self.models["pose"] = PoseEstimationDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames
                )

        if self.config.use_predictive_mask:
            self.models["predictive_mask"] = DepthDecoder(
                self.models["encoder"].num_ch_enc,
                self.config.scaling_factors,
                num_output_channels=(len(self.config.frame_ids) - 1),
            ).to(self.device)
            self.model_parameters += list(self.models["predictive_mask"].parameters())

        self.optimizer = optim.Adam(self.model_parameters, self.config.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, self.config.scheduler_step_size, 0.1
        )

        if self.config.weights_dir is not None:
            print(f"Using pretrained model weights from: {self.config.weights_dir}\n")
            self._load_model()

        print(f"Starting run: {self.config.run_name}\n")
        print("Training model: ", self.config.model_name, "\n")
        print("Models are saved in:  ", self.config.logs_dir, "\n")
        print("Current device:  ", self.device, "\n")

        self.dataset = (
            KITTIRAWDataset if self.config.dataset == "kitti" else KITTIOdomDataset
        )
        fpath = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))
            ),  # Поднимаемся на два уровня вверх
            "splits",
            self.config.split,
            "{}_files.txt",
        )
        print(__file__)
        print(fpath)

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_format = ".png" if self.config.png else ".jpg"

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

        if not self.config.no_ssim:
            self.ssim = SSIMLoss()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.config.scaling_factors:
            h = self.config.height // (2**scale)
            w = self.config.width // (2**scale)

            self.backproject_depth[scale] = BackprojectDepth(
                self.config.batch_size, h, w
            )
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.config.batch_size, h, w)
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

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()

    def train_epoch(self):
        """
        Single epoch with training and validation.
        """
        self.lr_scheduler.step()

        for model in self.models.values():
            model.train()

        self.writer.set_step(self.step)
        self.writer.add_scalar("epoch", self.epoch)

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            early_phase = (
                batch_idx % self.config.log_frequency == 0 and self.step < 2000
            )
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.writer.set_step(self.step)
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                for l, v in losses.items():
                    self.writer.add_scalar(l, v)
                self.val()

            self.step += 1

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            (self.num_total_steps / self.step - 1.0) * time_sofar
            if self.step > 0
            else 0
        )
        print_string = (
            "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}"
            + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        )
        print(
            print_string.format(
                self.epoch,
                batch_idx,
                samples_per_sec,
                loss,
                sec_to_hm_str(time_sofar),
                sec_to_hm_str(training_time_left),
            )
        )

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.config.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.config.frame_ids}
            else:
                pose_feats = {
                    f_i: inputs["color_aug", f_i, 0] for f_i in self.config.frame_ids
                }

            for f_i in self.config.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.config.pose_model_type == "separate_resnet":
                        pose_inputs = [
                            self.models["pose_encoder"](torch.cat(pose_inputs, 1))
                        ]
                    elif self.config.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                    )

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.config.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [
                        inputs[("color_aug", i, 0)]
                        for i in self.config.frame_ids
                        if i != "s"
                    ],
                    1,
                )

                if self.config.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.config.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.config.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.config.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i]
                    )

        return outputs

    def val(self):
        """Validate the model on a single minibatch"""
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.config.scaling_factors:
            disp = outputs[("disp", scale)]
            if self.config.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp,
                    [self.config.height, self.config.width],
                    mode="bilinear",
                    align_corners=False,
                )
                source_scale = 0

            _, depth = disp_to_depth(disp, self.config.min_depth, self.config.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.config.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.config.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0],
                        translation[:, 0] * mean_inv_depth[:, 0],
                        frame_id < 0,
                    )

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)]
                )
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T
                )

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                )

                if not self.config.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = inputs[
                        ("color", frame_id, source_scale)
                    ]

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(
            F.interpolate(
                depth_pred, [375, 1242], mode="bilinear", align_corners=False
            ),
            1e-3,
            80,
        )
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses"""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.config.pose_model_type == "shared":
            all_color_aug = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.config.frame_ids]
            )
            all_features = self.models["encoder"](all_color_aug)
            all_features = [
                torch.split(f, self.config.batch_size) for f in all_features
            ]

            features = {}
            for i, k in enumerate(self.config.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.config.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        params = {
            "options": self.config,
            "device": self.device,
            "num_scales": self.num_scales,
        }
        losses = calculate_losses(inputs, outputs, params)

        return outputs, losses

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
