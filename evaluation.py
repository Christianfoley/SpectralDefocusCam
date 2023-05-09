import numpy as np
import torch
import scipy
import os, glob
import matplotlib.pyplot as plt
import collections
import pandas as pd
import dataframe_image as dfi
import seaborn as sns
import imgkit

import lpips
import skimage.metrics as metrics
from sklearn.metrics.pairwise import cosine_similarity


import utils.helper_functions as helper


METRICS = ("mae", "mse", "cossim", "psnr", "ssim", "lpips")
STACK_DEPTH = (2, 3, 4, 5, 6)
PREDICTIONS = (
    "predictions/saved_model_ep640_2_testloss_0.0011793196899816394/",
    "predictions/saved_model_ep870_3_testloss_0.0004674650845117867/",
    "predictions/saved_model_ep660_5_testloss_0.0006101074395701289/",
    "predictions/saved_model_ep220_5_testloss_0.0010784256737679243/",
    "predictions/saved_model_ep1130_6_testloss_0.0003215452015865594/",
)
TABLE_DIR = "predictions/"
VISUALIZE_DIR = "predictions/"
ONE_NORMALIZE_VIS = True

# -------------- metric functions --------------#
# These functions accept float numpy arrays of shape c,y,x, the same dtype,
# and value_normed between 0 and 1


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
    return score


# structural (x-y) metrics
def get_mean_lpips_score(img1, img2, net="alex"):
    """alex for best forward scores, vgg closer to traditional percep loss (for opt)"""
    loss_fn = lpips.LPIPS(net=net)
    channels = img1.shape[-3]

    lpips_loss = []
    for i in range(0, channels // 3):
        a = torch.tensor(img1[..., 3 * i : 3 * i + 3, :, :])
        b = torch.tensor(img2[..., 3 * i : 3 * i + 3, :, :])
        lpips_loss.append(loss_fn(a, b).detach().numpy())
    return np.mean(np.asarray(lpips_loss))


def get_mean_psnr_score(img1, img2):
    slice_scores = []
    for i in range(img1.shape[-3]):
        score = metrics.peak_signal_noise_ratio(img1[i], img2[i], data_range=1)
        slice_scores.append(score)
    return np.mean(np.array(slice_scores))


def get_mean_ssim_score(img1, img2):
    slice_scores = []
    for i in range(img1.shape[-3]):
        score = metrics.structural_similarity(img1[i], img2[i], data_range=1)
        slice_scores.append(score)
    return np.mean(np.array(slice_scores))


# ----------- evaluation procedure ------------#
def get_score(metric, pred, sample):
    metrics_fns = {
        "mae": get_l1_score,
        "mse": get_l2_score,
        "cossim": get_cossim_score,
        "psnr": get_mean_psnr_score,
        "ssim": get_mean_ssim_score,
        "lpips": get_mean_lpips_score,
    }
    return metrics_fns[metric](pred, sample)


def get_metrics(preds_folder, metrics=METRICS, aggregate="mean"):
    pred_mats = glob.glob(os.path.join(preds_folder, "*.mat"))
    pred_mats = [p for p in pred_mats if "metrics.mat" not in p]

    scores = {metric: [] for metric in metrics}
    for i, mat_file in enumerate(pred_mats):
        mat = scipy.io.loadmat(mat_file)
        pred = helper.value_norm(mat["prediction"])
        sample = helper.value_norm(mat["sample"])

        for metric in metrics:
            scores[metric].append(get_score(metric, pred, sample))

    for metric in metrics:
        if aggregate == "mean":
            scores[metric] = np.mean(np.array(scores[metric]))
        elif aggregate == "sum":
            scores[metric] = sum(scores[metric])
        elif aggregate == "max":
            scores[metric] = max(scores[metric])

    scipy.io.savemat(os.path.join(preds_folder, "metrics.mat"), scores)
    return scores


def visualize_scores(all_scores, save_dir=VISUALIZE_DIR, stack_depths=STACK_DEPTH):
    if not isinstance(stack_depths, list):
        stack_depths = list(stack_depths)
    stack_depths = np.array(stack_depths)

    plt.figure(figsize=(14, 8), facecolor=(1, 1, 1))
    plt.title("Mean Metrics (one-normalized) vs Stack Depth")
    for metric, scores in all_scores.items():
        scores = np.array(scores)
        if ONE_NORMALIZE_VIS:
            scores = helper.value_norm(scores)

        if metric in ("mae", "mse", "lpips"):
            plt.plot(stack_depths, 1 - scores, label="1-" + metric)
        else:
            plt.plot(stack_depths, scores, label=metric)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "metrics_plot.png"))
    plt.show()


def generate_score_table(all_scores, save_dir=TABLE_DIR, stack_depths=STACK_DEPTH):
    if not isinstance(stack_depths, list):
        stack_depths = list(stack_depths)
    all_scores["depth"] = list(STACK_DEPTH)

    scores_df = pd.DataFrame(all_scores)
    cm = sns.light_palette("green", as_cmap=True)
    df_styled = scores_df.style.background_gradient(cmap=cm).highlight_max(
        axis="columns"
    )
    dfi.export(
        df_styled,
        os.path.join(save_dir, "metrics_table.png"),
        table_conversion="matplotlib",
    )


def main():
    scores = []
    for model_preds in PREDICTIONS:
        scores.append(get_metrics(model_preds))

    all_scores = collections.defaultdict(list)
    for scores_dict in scores:
        for key, value in scores_dict.items():
            all_scores[key].append(value)

    visualize_scores(all_scores, VISUALIZE_DIR, STACK_DEPTH)
    generate_score_table(all_scores, TABLE_DIR, STACK_DEPTH)


if __name__ == "__main__":
    main()
