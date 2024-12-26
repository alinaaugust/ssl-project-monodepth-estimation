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

    grad_img_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIMLoss(nn.Module):
    """Layer to compute the Structural Similarity (SSIM) loss between two images."""
    
    def __init__(self):
        super(SSIMLoss, self).__init__()

        self.mean_x_pool   = nn.AvgPool2d(3, 1)
        self.mean_y_pool   = nn.AvgPool2d(3, 1)
        self.var_x_pool    = nn.AvgPool2d(3, 1)
        self.var_y_pool    = nn.AvgPool2d(3, 1)
        self.cov_xy_pool   = nn.AvgPool2d(3, 1)

        self.pad_reflection = nn.ReflectionPad2d(1)

        self.epsilon1 = 0.01 ** 2
        self.epsilon2 = 0.03 ** 2

    def forward(self, img1, img2):

        img1 = self.pad_reflection(img1)
        img2 = self.pad_reflection(img2)

        mean_x = self.mean_x_pool(img1)
        mean_y = self.mean_y_pool(img2)

        var_x  = self.var_x_pool(img1 ** 2) - mean_x ** 2
        var_y  = self.var_y_pool(img2 ** 2) - mean_y ** 2
        cov_xy = self.cov_xy_pool(img1 * img2) - mean_x * mean_y

        num = (2 * mean_x * mean_y + self.epsilon1) * (2 * cov_xy + self.epsilon2)
        den = (mean_x ** 2 + mean_y ** 2 + self.epsilon1) * (var_x + var_y + self.epsilon2)

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



def compute_losses(inputs, outputs, opts):
    """
    Compute the reprojection and smoothness losses for a minibatch
    """

    losses = {}
    total_loss = 0

    for scale in opts['scales']:
        loss = 0
        reprojection_losses = []

        if opts['v1_multiscale']:
            source_scale = scale
        else:
            source_scale = 0

        disp = outputs[("disp", scale)]
        color = inputs[("color", 0, scale)]
        target = inputs[("color", 0, source_scale)]

        for frame_id in opts['frame_ids'][1:]:
            pred = outputs[("color", frame_id, scale)]
            reprojection_losses.append(compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)

        if not opts['disable_automasking']:
            identity_reprojection_losses = []
            for frame_id in opts['frame_ids'][1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if opts['avg_reprojection']:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                identity_reprojection_loss = identity_reprojection_losses

        elif opts['predictive_mask']:
            mask = outputs["predictive_mask"]["disp", scale]
            if not opts['v1_multiscale']:
                mask = F.interpolate(
                    mask, [opts['height'], opts['width']],
                    mode="bilinear", align_corners=False)

            reprojection_losses *= mask

            weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
            loss += weighting_loss.mean()

        if opts['avg_reprojection']:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not opts['disable_automasking']:
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=opts['device']) * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        if not opts['disable_automasking']:
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = smoothness_loss(norm_disp, color)

        loss += opts['disparity_smoothness'] * smooth_loss / (2 ** scale)
        total_loss += loss
        losses["loss/{}".format(scale)] = loss

    total_loss /= opts['num_scales']
    losses["loss"] = total_loss
    return losses