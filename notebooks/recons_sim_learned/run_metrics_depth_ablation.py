import sys, os, glob
import re, json
import scipy.io as io
import numpy as np
import pandas as pd
import tqdm

sys.path.insert(0, "../../")

import matplotlib.pyplot as plt
import dataset.precomp_dataset as ds
import utils.helper_functions as helper
import utils.metric_utils as metrics

TEST_DATA_PATH = "/home/cfoley/defocuscamdata/recons/model_ablation_test_set"

LEARNED_PREDS_PATH = (
    "/home/cfoley/defocuscamdata/recons/model_ablation_test_preds_learned"
)
FISTA_PREDS_PATH = "/home/cfoley/defocuscamdata/recons/model_ablation_test_preds_fista"
HANDSHAKE_PREDS_PATH = "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/results/handshake_fista"
IDEAL_FILTERS_PATH = "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/results/ideal_model_5_blurs_fista"
DIFUSER_PREDS_PATH = "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/results/diffuser_fista"


def get_idx(filename, pattern=r"idx(\d+)"):
    """Helper function for identifying index in dataset from filename"""
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return -1


def get_sample_names_fista(filenames):
    """Helper function to extract the source sample from the filenames, assuming fista recon format"""
    names = []
    for f in filenames:
        f = os.path.basename(f)
        names.append("_".join(f.split("_")[5:])[:-4])
    return names


def get_sample_names_learned(filenames):
    """Helper function to extract the source sample from the filenames, assuming lreaned recon format"""
    names = []
    for f in filenames:
        f = os.path.basename(f)
        names.append("_".join(f.split("_")[4:])[:-4])
    return names


# ---------------- FISTA RECONS -------------------- #
def separate_recons_by_tag(preds_path, name_tags: dict):
    """
    Separates files into a list of recons by model. name_tags maps model name to
    a list of "tags" which must be included in the filenames of the reconstruction
    from that model.
    """
    all_recons = glob.glob(os.path.join(preds_path, "*.npy"))

    # Separate files uniquely by model: filenames must contain all tags
    files = {k: None for k in name_tags.keys()}

    for name, tags in name_tags.items():
        files[name] = sorted(
            [fname for fname in all_recons if all([t in fname for t in tags])]
        )
    return files


def build_score_table(preds, pred_names):
    """
    Returns a dataframe with a header row of metric names, then following rows containing
    scores by each metric
    """
    metric_names = ["mse", "cossim", "psnr", "ssim"]
    df = pd.DataFrame(columns=metric_names)

    test_data_files = glob.glob(os.path.join(TEST_DATA_PATH, "*.mat"))
    sample_names = [os.path.basename(f)[:-4] for f in test_data_files]
    unpredicted = set(sample_names).symmetric_difference(set(pred_names))

    preds, test_data_files = sorted(preds), sorted(test_data_files)
    preds_idx = 0
    for _, sample in tqdm.tqdm(list(enumerate(test_data_files))):
        if os.path.basename(sample)[:-4] in unpredicted:
            continue
        pred, gt = (
            np.load(preds[preds_idx]).transpose(2, 0, 1),
            io.loadmat(sample)["image"],
        )

        # ------ Normalize into same region ------ #
        pred = (pred - np.mean(pred)) / np.std(pred)
        gt = (gt - np.mean(gt)) / np.std(gt)

        sample_scores = {}
        for metric in metric_names:
            score = metrics.get_score(metric, pred, gt)
            sample_scores[metric] = score

        df.loc[preds_idx] = sample_scores
        preds_idx += 1

    df = df.mean(axis=0)
    return df


def build_model_score_tables(preds_path, name_function, name_tags):
    """Returns dictionary of score tables for each model"""
    model_scores = {}
    recon_files = separate_recons_by_tag(preds_path, name_tags)
    for model_name, preds in recon_files.items():
        model_scores[model_name] = build_score_table(preds, name_function(preds))
    return model_scores


def main():
    # ---------- Defocuscam methods & blur ablation ---------- #
    # name_tags = {
    #     "stack1_blurry": ["_1_", "blurry-True"],
    #     "stack1_sharp": ["_1_", "blurry-False"],
    #     "stack2": ["_2_"],
    #     "stack3": ["_3_"],
    #     "stack5": ["_5_"],
    # }

    # fista_metrics = build_model_score_tables(
    #     FISTA_PREDS_PATH, get_sample_names_fista, name_tags
    # )

    # fista_metrics = {key: df.to_dict() for key, df in fista_metrics.items()}
    # with open("fista_scores.json", "w") as f:
    #     f.write(json.dumps(fista_metrics, indent=4, separators=(",", ": ")))

    # learned_metrics = build_model_score_tables(
    #     LEARNED_PREDS_PATH, get_sample_names_learned, name_tags
    # )

    # learned_metrics = {key: df.to_dict() for key, df in learned_metrics.items()}
    # with open("learned_scores.json", "w") as f:
    #     f.write(json.dumps(learned_metrics, indent=4, separators=(",", ": ")))

    # -------------- Handshake -------------- #
    name_tags = {
        "handshake_9": ["_9_", "blurry-False"],
    }
    handshake_metrics = build_model_score_tables(
        HANDSHAKE_PREDS_PATH, get_sample_names_fista, name_tags
    )
    handshake_metrics = {key: df.to_dict() for key, df in handshake_metrics.items()}
    with open("handshake_scores.json", "w") as f:
        f.write(json.dumps(handshake_metrics, indent=4, separators=(",", ": ")))

    # -------------- Diffuser -------------- #
    name_tags = {
        "diffusercam": ["_1_", "blurry-False"],
    }
    diffusercam_metrics = build_model_score_tables(
        DIFUSER_PREDS_PATH, get_sample_names_fista, name_tags
    )
    diffusercam_metrics = {key: df.to_dict() for key, df in diffusercam_metrics.items()}
    with open("diffuser_scores.json", "w") as f:
        f.write(json.dumps(diffusercam_metrics, indent=4, separators=(",", ": ")))

    # -------------- Ideal filter alignments -------------- #
    name_tags = {
        "ideal_filters_5": ["_5_", "blurry-False"],
    }
    ideal_filters_metrics = build_model_score_tables(
        IDEAL_FILTERS_PATH, get_sample_names_fista, name_tags
    )
    ideal_filters_metrics = {
        key: df.to_dict() for key, df in ideal_filters_metrics.items()
    }
    with open("ideal_filters_scores.json", "w") as f:
        f.write(json.dumps(ideal_filters_metrics, indent=4, separators=(",", ": ")))


if __name__ == "__main__":
    main()
