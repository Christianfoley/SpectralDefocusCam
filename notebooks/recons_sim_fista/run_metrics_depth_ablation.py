# %%
import sys, os, glob
import re
import numpy as np
import pandas as pd
import tqdm

sys.path.insert(0, "../../")


import dataset.precomp_dataset as ds
import utils.helper_functions as helper
import utils.metric_utils as metrics

# %%
sample_data_path = "/home/cfoley_waller/10tb_extension/defocam/defocuscamdata/sample_data_preprocessed/sample_data_preprocessed_lsi_02_07"
train_loader, val_loader, test_loader = ds.get_data_precomputed(
    1, (0.7, 0.15, 0.15), sample_data_path
)

# %%
fista_folder = (
    "/home/cfoley_waller/10tb_extension/defocam/defocuscamdata/recons/sim_fista_03_07"
)
learned_folder = ""


def get_idx(filename, pattern=r"idx(\d+)"):
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return -1  # Return -1 if no match found


all_recons = glob.glob(os.path.join(fista_folder, "*.npy"))
stack2 = sorted(
    [filename for filename in all_recons if "2stack" in filename], key=get_idx
)
stack3 = sorted(
    [filename for filename in all_recons if "3stack" in filename], key=get_idx
)
stack5 = sorted(
    [filename for filename in all_recons if "5stack" in filename], key=get_idx
)


# %%
def build_score_table(test_loader, preds):
    """
    Returns a dataframe with a header row of metric names, then following rows containing
    scores by each metric
    """
    metric_names = ["mse", "cossim", "psnr", "ssim"]
    df = pd.DataFrame(columns=metric_names)

    for i, sample in tqdm.tqdm(list(enumerate(test_loader))):
        pred, gt = np.load(preds[i]).transpose(2, 0, 1), sample["image"][0].numpy()
        pred = (pred - np.mean(pred)) / np.std(pred)

        sample_scores = {}
        for metric in metric_names:
            score = metrics.get_score(metric, pred, gt)
            sample_scores[metric] = score

        df.loc[i] = sample_scores

    return df


# %%
stack2_scores = build_score_table(test_loader, stack2)
stack3_scores = build_score_table(test_loader, stack3)
stack5_scores = build_score_table(test_loader, stack5)
# %%
stack2_scores.mean(axis=0)
# %%
stack3_scores.mean(axis=0)
# %%
stack5_scores.mean(axis=0)

# %%
