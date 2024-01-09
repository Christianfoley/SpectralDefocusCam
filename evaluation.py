import numpy as np
import os
import matplotlib.pyplot as plt
import collections
import pandas as pd
import dataframe_image as dfi
import seaborn as sns

import utils.helper_functions as helper
import SpectralDefocusCam.utils.metric_utils as metric_utils


METRICS = ("mae", "mse", "cossim", "psnr", "ssim", "lpips")
STACK_DEPTH = (4, 5)
PREDICTIONS = (
    "../defocuscamdata/predictions/models/checkpoint_4_5_True_symmetric_False/2023_05_09_20_44_31/predictions",
    "../home/cfoley_waller/defocam/defocuscamdata/models/checkpoint_4_4_True_symmetric_False/2023_05_09_14_08_32/",
)
TABLE_DIR = "predictions/5_10_2023_stack45"
VISUALIZE_DIR = "predictions/5_10_2023_stack45"
ONE_NORMALIZE_VIS = True

# -------------- metric functions --------------#
# These functions accept float numpy arrays of shape c,y,x, the same dtype,
# and value_normed between 0 and 1


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
        scores.append(metric_utils.get_metrics(model_preds, METRICS))

    all_scores = collections.defaultdict(list)
    for scores_dict in scores:
        for key, value in scores_dict.items():
            all_scores[key].append(value)

    visualize_scores(all_scores, VISUALIZE_DIR, STACK_DEPTH)
    generate_score_table(all_scores, TABLE_DIR, STACK_DEPTH)


if __name__ == "__main__":
    main()
