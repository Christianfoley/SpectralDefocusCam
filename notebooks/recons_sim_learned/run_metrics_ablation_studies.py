#%%
import sys, os, glob
import re, json
import scipy.io as io
import numpy as np
import pandas as pd
import tqdm
import multiprocessing as mp

sys.path.insert(0, "../../")

import matplotlib.pyplot as plt
import dataset.precomp_dataset as ds
import utils.helper_functions as helper
import utils.metric_utils as metrics

TEST_DATA_PATH = "/home/cfoley/defocuscamdata/recons/model_ablation_test_set"

LEARNED_CONDUNET_PREDS_PATH = (
    "/home/cfoley/defocuscamdata/recons/model_ablation_test_preds_learned"
)
FIRSTLAST_PREDS_PATH_LEARNED = (
    "/home/cfoley/defocuscamdata/recons/model_ablation_test_preds_learned_testfirstlast"
)
CLASSICAL_UNET_PREDS_PATH = (
    "/home/cfoley/defocuscamdata/recons/model_ablation_test_preds_learned_testclassicalunet"
)
FISTA_PREDS_PATH = "/home/cfoley/defocuscamdata/recons/model_ablation_test_preds_fista"
HANDSHAKE_PREDS_PATH = "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/results/handshake_new_random_fista"
IDEAL_FILTERS_PATH = "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/results/ideal_model_5_blurs_fista"
DIFUSER_PREDS_PATH = "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/results/diffuser_fista_exp_filter"
NOISE_PREDS_PATH_FISTA = (
    "/home/cfoley/defocuscamdata/recons/noise_ablation_maskandshot_preds_fista"
)
NOISE_PREDS_PATH_LEARNED_CONDFSTLST = (
    "/home/cfoley/defocuscamdata/recons/noise_ablation_maskandshot_preds_learned_fstlst"
)
NOISE_PREDS_PATH_LEARNED_COND = (
    "/home/cfoley/defocuscamdata/recons/noise_ablation_maskandshot_preds_learned_condunet_2meas"
)
NOISE_PREDS_PATH_LEARNED_CLASSICAL = (
    "/home/cfoley/defocuscamdata/recons/noise_ablation_maskandshot_preds_learned_classical_unet"
)


def get_idx(filename, pattern=r"idx(\d+)"):
    """Helper function for identifying index in dataset from filename"""
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return -1


def get_sample_names_5(filenames):
    """Helper function to extract the source sample from the filenames, assuming fista recon format"""
    names = []
    for f in filenames:
        f = os.path.basename(f)
        names.append("_".join(f.split("_")[5:])[:-4])
    return names


def get_sample_names_4(filenames):
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


def calculate_sample_scores(total, idx, pred_file, gt_file):
    """ Helper for multiprocessing """
    sample_scores = {}
    metric_names = ["mse", "cossim", "psnr", "ssim"]

    pred = np.load(pred_file).transpose(2, 0, 1)
    gt = io.loadmat(gt_file)["image"]

    # remove extra spectral dims if there are any
    if pred.shape[0] == 32:
        pred = pred[1:-1]

    # Normalize images
    try:
        pred = (pred - np.mean(pred)) / np.std(pred)
        gt = (gt - np.mean(gt)) / np.std(gt)
    except Exception as e:
        print(f"Error {os.path.basename(pred_file)} : {e} ... Skipping.")
        return None

    if np.any(np.isnan(pred)) or np.any(np.isnan(gt)):
        print(f"NaN {os.path.basename(pred_file)}... Skipping.")
        return None

    for metric in metric_names:
        score = metrics.get_score(metric, pred, gt)
        if np.isnan(score):
            print(f"NaN {metric} {os.path.basename(pred_file)}... Skipping.")
            return None
        sample_scores[metric] = score

    print(f"Processed {idx} / {total}", end="\r")
    return sample_scores

def build_score_table(preds, pred_names):
    """
    Returns a dataframe with a header row of metric names, then following rows containing
    scores by each metric
    """
    metric_names = ["mse", "cossim", "psnr", "ssim"]
    df = pd.DataFrame(columns=metric_names)

    test_data_files = sorted(glob.glob(os.path.join(TEST_DATA_PATH, "*.mat")))
    sample_names = sorted([os.path.basename(f)[:-4] for f in test_data_files])
    pred_names, preds = sorted(pred_names), sorted(preds)

    assert len(set(pred_names)) == len(pred_names), "Prediction patches not unique!"

    args_list = []
    pred_idx = 0
    for i, gt_file in tqdm.tqdm(list(enumerate(test_data_files)), desc="Collecting "): # maps files to eachother correctly
        if pred_idx >= len(pred_names) or sample_names[i] != pred_names[pred_idx]:
            continue

        # safety check
        assert pred_names[pred_idx] == sample_names[i], f"Not equal - {pred_names[pred_idx]} : {sample_names[i]}"
        args_list.append([pred_idx, preds[pred_idx], gt_file])
        pred_idx += 1

    with mp.Pool() as pool:
        num = len(args_list)
        sample_scores_list = [pool.apply_async(calculate_sample_scores, [num] + args) for args in args_list]
        results = [result.get() for result in sample_scores_list if result.get() is not None]

    for i, sample_scores in enumerate(results):
        df.loc[i] = sample_scores

    df = df.mean(axis=0)
    return df.to_dict()


def build_model_score_tables(preds_path, name_function, name_tags):
    """Returns dictionary of score tables for each model"""
    model_scores = {}
    recon_files = separate_recons_by_tag(preds_path, name_tags)
    for model_name, preds in recon_files.items():
        model_scores[model_name] = build_score_table(preds, name_function(preds))
    return model_scores


def main():
    # # ---------- Defocuscam methods & blur ablation ---------- #
    # name_tags = {
    #     "stack1_blurry": ["_1_", "blurry-True"],
    #     "stack1_sharp": ["_1_", "blurry-False"],
    #     "stack2": ["_2_"],
    #     "stack3": ["_3_"],
    #     "stack5": ["_5_"],
    # }

    # fista_metrics = build_model_score_tables(
    #     FISTA_PREDS_PATH, get_sample_names_5, name_tags
    # )

    # with open("fista_scores.json", "w") as f:
    #     f.write(json.dumps(fista_metrics, indent=4, separators=(",", ": ")))

    # learned_metrics = build_model_score_tables(
    #     LEARNED_PREDS_PATH, get_sample_names_4, name_tags
    # )

    # with open("learned_scores.json", "w") as f:
    #     f.write(json.dumps(learned_metrics, indent=4, separators=(",", ": ")))

    # # -------------- Handshake -------------- #
    # name_tags = {
    #     "handshake_5": ["_5_", "blurry-False"],
    # }
    # handshake_metrics = build_model_score_tables(
    #     HANDSHAKE_PREDS_PATH, get_sample_names_5, name_tags
    # )
    # with open("handshake_scores_new.json", "w") as f:
    #     f.write(json.dumps(handshake_metrics, indent=4, separators=(",", ": ")))

    # # # -------------- Diffuser -------------- #
    # name_tags = {
    #     "diffusercam": ["_1_", "blurry-False"],
    # }
    # diffusercam_metrics = build_model_score_tables(
    #     DIFUSER_PREDS_PATH, get_sample_names_5, name_tags
    # )
    # with open("diffuser_scores.json", "w") as f:
    #     f.write(json.dumps(diffusercam_metrics, indent=4, separators=(",", ": ")))

    # # -------------- Ideal filter alignments -------------- #
    # name_tags = {
    #     "ideal_filters_5": ["_5_", "blurry-False"],
    # }
    # ideal_filters_metrics = build_model_score_tables(
    #     IDEAL_FILTERS_PATH, get_sample_names_5, name_tags
    # )
    # with open("ideal_filters_scores.json", "w") as f:
    #     f.write(json.dumps(ideal_filters_metrics, indent=4, separators=(",", ": ")))

    # # -------------- Noise ablation experiments -------------- #
    # name_tags = {
    #     "noise_1e-1": ["(0.1, True, True)"],
    #     "noise_5e-2": ["(0.05, True, True)"],
    #     "noise_1e-2": ["(0.01, True, True)"],
    #     "noise_1e-3": ["(0.001, True, True)"],
    #     "noise_1e-4": ["(0.0001, True, True)"],
    #     "noise_1e-5": ["(1e-05, True, True)"],
    # }
    # noise_fista_metrics = build_model_score_tables(
    #     NOISE_PREDS_PATH_FISTA, get_sample_names_5, name_tags
    # )
    # with open("fista_maskandshot_noise_scores.json", "w") as f:
    #     f.write(json.dumps(noise_fista_metrics, indent=4, separators=(",", ": ")))

    # name_tags = {
    #     "noise_1e-1": ["noise-0.1"],
    #     "noise_5e-2": ["noise-0.05"],
    #     "noise_1e-2": ["noise-0.01"],
    #     "noise_1e-3": ["noise-0.001"],
    #     "noise_1e-4": ["noise-0.0001"],
    #     "noise_1e-5": ["noise-1e-05"],
    # }

    # noise_learned_metrics_cond_fstlst = build_model_score_tables(
    #     NOISE_PREDS_PATH_LEARNED_CONDFSTLST, get_sample_names_5, name_tags
    # )
    # with open("learned_fstlst_maskandshot_noise_scores.json", "w") as f:
    #     f.write(json.dumps(noise_learned_metrics, indent=4, separators=(",", ": ")))

    # name_tags = {
    #     "noise_1e-1": ["noise-0.1"],
    #     "noise_5e-2": ["noise-0.05"],
    #     "noise_1e-2": ["noise-0.01"],
    #     "noise_1e-3": ["noise-0.001"],
    #     "noise_1e-4": ["noise-0.0001"],
    #     "noise_1e-5": ["noise-1e-05"],
    # }

    # noise_learned_metrics_condunet = build_model_score_tables(
    #     NOISE_PREDS_PATH_LEARNED_COND, get_sample_names_5, name_tags
    # )
    # with open("learned_cond_maskandshot_noise_scores.json", "w") as f:
    #     f.write(json.dumps(noise_learned_metrics_condunet, indent=4, separators=(",", ": ")))

    name_tags = {
        "noise_1e-1": ["noise-0.1"],
        "noise_5e-2": ["noise-0.05"],
        "noise_1e-2": ["noise-0.01"],
        "noise_1e-3": ["noise-0.001"],
        "noise_1e-4": ["noise-0.0001"],
        "noise_1e-5": ["noise-1e-05"],
    }

    noise_learned_metrics_classicalunet = build_model_score_tables(
        NOISE_PREDS_PATH_LEARNED_CLASSICAL, get_sample_names_5, name_tags
    )
    with open("learned_classical_maskandshot_noise_scores.json", "w") as f:
        f.write(json.dumps(noise_learned_metrics_classicalunet, indent=4, separators=(",", ": ")))

    # -------------- Testing first-last conditioning model -------------- #
    # name_tags = {
    #     "stack1_blurry": ["_1_", "blurry-True"],
    #     "stack1_sharp": ["_1_", "blurry-False"],
    #     "stack2": ["_2_"],
    #     "test_3_firstlastonly": ["_3_", "blurry-False"],
    #     "test_5_firstlastonly": ["_5_", "blurry-False"],
    # }

    # noise_learned_metrics = build_model_score_tables(
    #     FIRSTLAST_PREDS_PATH_LEARNED, get_sample_names_4, name_tags
    # )
    # with open("learned_scores_firstlastonly.json", "w") as f:
    #     f.write(json.dumps(noise_learned_metrics, indent=4, separators=(",", ": ")))

    # -------------- Testing no conditioning model -------------- #
    # name_tags = {
    #     "stack1_blurry": ["_1_", "blurry-True"],
    #     "stack1_sharp": ["_1_", "blurry-False"],
    #     "stack2": ["_2_"],
    #     "test_3_firstlastonly": ["_3_", "blurry-False"],
    #     "test_5_firstlastonly": ["_5_", "blurry-False"],
    # }

    # noise_learned_metrics = build_model_score_tables(
    #     CLASSICAL_UNET_PREDS_PATH, get_sample_names_4, name_tags
    # )
    # with open("classical_unet_scores.json", "w") as f:
    #     f.write(json.dumps(noise_learned_metrics, indent=4, separators=(",", ": ")))

if __name__ == "__main__":
    main()
# %%
