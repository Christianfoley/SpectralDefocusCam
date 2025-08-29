import numpy as np
import glob
import os

import skimage.metrics as metrics
from sklearn.metrics.pairwise import cosine_similarity
import scipy

import defocuscam.utils.helper_functions as helper


# absolute metrics
def get_l1_score(img1, img2, axis=None):
    mse = np.mean((np.abs(img1 - img2)), axis=axis)
    return mse


def get_l2_score(img1, img2, axis=None):
    mse = np.mean(((img1 - img2) ** 2), axis=axis)
    return mse


def get_cossim_score(img1, img2):
    img1, img2 = img1.flatten(), img2.flatten()
    img1, img2 = np.expand_dims(img1, 0), np.expand_dims(img2, 0)
    score = cosine_similarity(img1, img2)
    return score.item()


# structural (x-y) metrics
def get_mean_psnr_score(img1, img2):
    mse = get_l2_score(img1, img2)
    psnr = 20 * np.log10(1 / np.sqrt(mse))
    return psnr


def get_mean_ssim_score(img1, img2):
    data_range = max(np.max(img1) - np.min(img1), np.max(img2) - np.min(img2))
    score = metrics.structural_similarity(img1, img2, data_range=data_range)
    return score


# ----------- evaluation procedure ------------#
def get_score(metric, pred, sample):
    metrics_fns = {
        "mae": get_l1_score,
        "mse": get_l2_score,
        "cossim": get_cossim_score,
        "psnr": get_mean_psnr_score,
        "ssim": get_mean_ssim_score,
    }
    return metrics_fns[metric](pred, sample)


def get_metrics(preds_folder, metrics_list, aggregate="mean"):
    pred_mats = glob.glob(os.path.join(preds_folder, "*.mat"))
    pred_mats = [p for p in pred_mats if "metrics.mat" not in p]

    scores = {metric: [] for metric in metrics_list}
    for i, mat_file in enumerate(pred_mats):
        mat = scipy.io.loadmat(mat_file)
        pred = helper.value_norm(mat["prediction"])
        sample = helper.value_norm(mat["sample"])

        for metric in metrics_list:
            scores[metric].append(get_score(metric, pred, sample))

    for metric in metrics_list:
        if aggregate == "mean":
            scores[metric] = np.mean(np.array(scores[metric]))
        elif aggregate == "sum":
            scores[metric] = sum(scores[metric])
        elif aggregate == "max":
            scores[metric] = max(scores[metric])

    scipy.io.savemat(os.path.join(preds_folder, "metrics.mat"), scores)
    return scores
