import numpy as np


def depth_errors(ground_truth, prediction):
    """
    Calculate error metrics between predicted and ground truth depth maps.
    """
    ratio = np.maximum(ground_truth / prediction, prediction / ground_truth)
    delta1 = (ratio < 1.25).mean()
    delta2 = (ratio < 1.25**2).mean()
    delta3 = (ratio < 1.25**3).mean()

    mse = (ground_truth - prediction) ** 2
    rmse = np.sqrt(np.mean(mse))

    log_diff = (np.log(ground_truth) - np.log(prediction)) ** 2
    rmse_log = np.sqrt(log_diff.mean())

    abs_rel_error = np.mean(np.abs(ground_truth - prediction) / ground_truth)

    sq_rel_error = np.mean(mse / ground_truth)

    return abs_rel_error, sq_rel_error, rmse, rmse_log, delta1, delta2, delta3
