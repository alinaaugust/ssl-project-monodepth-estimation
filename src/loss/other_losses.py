import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def smoothness_loss(disparity, image):
    """
    Calculate the smoothness loss for a disparity map with edge-aware weighting using the color image.
    """

    grad_disp_x = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
    grad_disp_y = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, :])

    grad_img_x = torch.mean(
        torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True
    )

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIMLoss(nn.Module):
    """Layer to compute the Structural Similarity (SSIM) loss between two images."""

    def __init__(self):
        super(SSIMLoss, self).__init__()

        self.mean_x_pool = nn.AvgPool2d(3, 1)
        self.mean_y_pool = nn.AvgPool2d(3, 1)
        self.var_x_pool = nn.AvgPool2d(3, 1)
        self.var_y_pool = nn.AvgPool2d(3, 1)
        self.cov_xy_pool = nn.AvgPool2d(3, 1)

        self.pad_reflection = nn.ReflectionPad2d(1)

        self.epsilon1 = 0.01**2
        self.epsilon2 = 0.03**2

    def forward(self, img1, img2):

        img1 = self.pad_reflection(img1)
        img2 = self.pad_reflection(img2)

        mean_x = self.mean_x_pool(img1)
        mean_y = self.mean_y_pool(img2)

        var_x = self.var_x_pool(img1**2) - mean_x**2
        var_y = self.var_y_pool(img2**2) - mean_y**2
        cov_xy = self.cov_xy_pool(img1 * img2) - mean_x * mean_y

        num = (2 * mean_x * mean_y + self.epsilon1) * (2 * cov_xy + self.epsilon2)
        den = (mean_x**2 + mean_y**2 + self.epsilon1) * (var_x + var_y + self.epsilon2)

        return torch.clamp((1 - num / den) / 2, 0, 1)


def compute_reprojection_loss(pred, target, ssim_fn, no_ssim=False):
    """
    Computes reprojection loss between predicted and target images.

    Args:
        pred (Tensor): Predicted image batch.
        target (Tensor): Target image batch.
        ssim_fn (function): Function to compute SSIM loss.
        no_ssim (bool): Flag to decide if SSIM loss should be used (default is False).

    Returns:
        Tensor: The reprojection loss value.
    """

    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if no_ssim:
        reprojection_loss = l1_loss
    else:
        ssim_loss = ssim_fn(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


def calculate_losses(data_inputs, model_outputs, settings):
    """
    Calculate reprojection and smoothness losses for a batch of data.

    Args:
        data_inputs (dict): Contains input data (e.g., images at different scales).
        model_outputs (dict): Contains outputs from the model (e.g., disparity maps).
        settings (dict): A dictionary with the following required fields:
            - "config": object containing configuration like scales and frame_ids
            - "device": computation device ("cuda" or "cpu")
            - "num_scales": number of scales for averaging the total loss
            - "reprojection_loss_fn": function to compute reprojection loss
            - "smoothness_loss_fn": function to compute disparity smoothness loss

    Returns:
        dict: A dictionary containing the per-scale losses and the overall loss.
    """
    loss_details = {}
    cumulative_loss = 0

    for scale_level in settings["config"].scales:
        current_loss = 0
        reprojection_loss_list = []

        # Determine the appropriate source scale
        if settings["config"].v1_multiscale:
            source_scale = scale_level
        else:
            source_scale = 0

        disparity = model_outputs[("disp", scale_level)]
        image_at_scale = data_inputs[("color", 0, scale_level)]
        target_image = data_inputs[("color", 0, source_scale)]

        # Compute reprojection losses for each frame
        for frame_index in settings["config"].frame_ids[1:]:
            predicted_image = model_outputs[("color", frame_index, scale_level)]
            reprojection_loss_list.append(
                settings["reprojection_loss_fn"](predicted_image, target_image)
            )

        reprojection_losses = torch.cat(reprojection_loss_list, dim=1)

        # Handle identity reprojection loss if automasking is enabled
        if not settings["config"].disable_automasking:
            identity_loss_list = []
            for frame_index in settings["config"].frame_ids[1:]:
                identity_prediction = data_inputs[("color", frame_index, source_scale)]
                identity_loss_list.append(
                    settings["reprojection_loss_fn"](identity_prediction, target_image)
                )

            identity_losses = torch.cat(identity_loss_list, dim=1)

            if settings["config"].avg_reprojection:
                identity_reprojection_loss = identity_losses.mean(1, keepdim=True)
            else:
                identity_reprojection_loss = identity_losses

        elif settings["config"].predictive_mask:
            mask = model_outputs["predictive_mask"]["disp", scale_level]
            if not settings["config"].v1_multiscale:
                mask = F.interpolate(
                    mask,
                    [settings["config"].height, settings["config"].width],
                    mode="bilinear",
                    align_corners=False,
                )

            reprojection_losses *= mask

            # Add weighting loss to encourage masks toward 1
            mask_loss = 0.2 * nn.BCELoss()(
                mask, torch.ones_like(mask).to(settings["device"])
            )
            current_loss += mask_loss.mean()

        if settings["config"].avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not settings["config"].disable_automasking:
            # Break ties with small random noise
            identity_reprojection_loss += (
                torch.randn(identity_reprojection_loss.shape, device=settings["device"])
                * 1e-5
            )

            combined_losses = torch.cat(
                (identity_reprojection_loss, reprojection_loss), dim=1
            )
        else:
            combined_losses = reprojection_loss

        # Determine the minimum loss to optimize
        if combined_losses.shape[1] == 1:
            optimized_loss = combined_losses
        else:
            optimized_loss, indices = torch.min(combined_losses, dim=1)

        if not settings["config"].disable_automasking:
            model_outputs[f"identity_selection/{scale_level}"] = (
                indices > identity_reprojection_loss.shape[1] - 1
            ).float()

        current_loss += optimized_loss.mean()

        # Compute smoothness loss for the disparity
        mean_disparity = disparity.mean(2, True).mean(3, True)
        normalized_disparity = disparity / (mean_disparity + 1e-7)
        smoothness_loss = settings["smoothness_loss_fn"](
            normalized_disparity, image_at_scale
        )

        current_loss += (
            settings["config"].disparity_smoothness * smoothness_loss / (2**scale_level)
        )
        cumulative_loss += current_loss

        loss_details[f"loss/{scale_level}"] = current_loss

    # Average loss over all scales
    cumulative_loss /= settings["num_scales"]
    loss_details["total_loss"] = cumulative_loss
    return loss_details
