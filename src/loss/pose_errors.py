import numpy as np


def pose_errors(ground_truth, predicted_poses):
    """
    Compute Absolute Trajectory Error (ATE) between ground truth and predicted poses.
    """
    # Align the first frames of ground truth and predictions
    translation_offset = ground_truth[0] - predicted_poses[0]
    aligned_predictions = predicted_poses + translation_offset[None, :]

    # Optimize scaling factor
    scaling_factor = np.sum(ground_truth * aligned_predictions) / np.sum(
        aligned_predictions**2
    )
    alignment_difference = aligned_predictions * scaling_factor - ground_truth
    rmse = np.sqrt(np.sum(alignment_difference**2)) / ground_truth.shape[0]

    return rmse
