# ------------------------------------------- #
#  Utilities for runnning simulation studies  #
#  on our 849-sample test set under varying   #
#  conditions.                                #
# ------------------------------------------- #

import glob
import os
import pathlib
import time
import torch
import json
import scipy.io as io
import multiprocessing as mp
import tqdm
import wandb
from typing import Any, Optional
from pprint import pprint


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import utils.helper_functions as helper
import utils.metric_utils as metrics
from models.ensemble import SSLSimulationModel
from models.get_model import get_model
import numpy as np
import dataset.dataset as ds

from torchvision import transforms
from torch.utils.data import DataLoader

# Set the torch seed
torch.manual_seed(6.626)

METRICS = ["mse", "cossim", "psnr", "ssim"]


def run_study_sweep(
    study_name: str,
    config_paths: list[str],
    overrides: Optional[list[dict]] = None,
    project_name: str = "SpectralDefocusCam",
    overwrite_existing_recons: bool | list[bool] = False,
    overwrite_existing_metrics: bool | list[bool] = False,
    study_name_suffixes: Optional[list[str]] = None,
):
    """
    Run a full study sweep over a set of config_paths, creating a WandB run for each config, generating
    all reconstructions according to the config's specification, and computing metrics for that config's
    reconstructions.

    Parameters
    ----------
    study_name : str
        Name of the study, used as a suffix in WandB
    config_paths : list[str]
        List of paths to the config files which describe the sweep
    overrides : Optional[list[dict]], None
        List of override dictionaries (e.g. {"param.subparam.subsubparam": "override_value"}) to apply
        to the parameters of each config at the corresponding index
    project_name : str, "SpectralDefocusCam
        Name of the project in WandB
    overwrite_existing_recons : bool | list[bool], Fasle
        Whether to overwrite reconstruction files in existing paths, or skip them if they exist. If
        a bool, applies to all runs. If a list of bools, applies to the run at the corresponding
        index
    overwrite_existing_metrics : bool | list[bool], False
        Whether to overwrite metrics files in existing paths, or skip them if they exist. If a
        bool, applies to all runs. If a list of bools, applies to the run at the corresponding
        index
    study_name_suffixes : Optional[list[str]], None
        A list of suffixes for the name of each study, added at the end of the study name. This
        is helpful to distinguish runs in WandB if different studies point to the same config
        path but have different overrides

    """
    # Do some initial cleansing to make sure the run is set up right
    if isinstance(overwrite_existing_recons, bool):
        overwrite_existing_recons = [overwrite_existing_recons] * len(config_paths)
    if isinstance(overwrite_existing_metrics, bool):
        overwrite_existing_metrics = [overwrite_existing_metrics] * len(config_paths)
    if overrides is None:
        overrides = [{}] * len(config_paths)
    if study_name_suffixes is None:
        study_name_suffixes = [""] * len(config_paths)
    assert (
        len(config_paths)
        == len(overwrite_existing_metrics)
        == len(overwrite_existing_recons)
        == len(overrides)
    ), (
        "Unmatched arguments for sweep! "
        "Expected all sweep arguments to be the same length, "
        f"but got: config_paths={len(config_paths)}, "
        f"overwrite_existing_metrics={len(overwrite_existing_metrics)}, "
        f"overwrite_existing_recons={len(overwrite_existing_recons)}, "
        f"overrides={len(overrides)}"
    )

    print(f"Collected {len(config_paths)} configs:")
    for i, config_path in enumerate(config_paths):
        print("\t", os.path.abspath(config_path))
        print(
            f"\t\toverwrite_existing_metrics={overwrite_existing_metrics[i]}, "
            f"overwrite_existing_recons={overwrite_existing_recons[i]}, "
            f"overrides={overrides[i]}"
        )
    print("\n")

    print("Beginning run")
    start = time.time()
    for (
        config_path,
        override_params,
        overwrite_recons,
        overwrite_metrics,
        suffix,
    ) in zip(
        config_paths,
        overrides,
        overwrite_existing_recons,
        overwrite_existing_metrics,
        study_name_suffixes,
    ):
        if not os.path.exists(config_path):
            print(f"Config {config_path} not found! Continuing...")

        run_name = f"{study_name}_{pathlib.Path(config_path).stem}"
        wandb_run = wandb.init(
            project=project_name,
            name=f"{run_name}_{suffix}" if len(suffix) else run_name,
            config=_override_config_parameters(
                helper.read_config(config_path), override_params
            ),
        )

        try:
            reconstructed_files = run_reconstruction_grid(
                config_path, overwrite_existing=overwrite_recons
            )
            wandb.log({"Num_valid_recons": len(reconstructed_files)})
        except Exception as e:
            print(f"Reconstructions failed: {str(e)}")

        try:
            metrics_path = compute_metrics(
                config_path, overwrite_existing=overwrite_metrics
            )
            with open(metrics_path, "r") as f:
                wandb.log(json.load(f))
        except Exception as e:
            print(f"Metrics computation failed:{str(e)}")

        wandb_run.finish()

    print("\n")
    print("Done!")
    print(f"Total time: {time.time() - start:.2f} seconds.")


def run_reconstruction_grid(
    config_path: str, overwrite_existing=False, override_params={}
) -> list[str]:
    """
    Given a complete configuration file, set up and run an experiment of reconstructions according to
    the model and data specified in the config, saving and returning paths to the saved reconstructions

    Parameters:
    -----------
    config_path : str
        Path to the config `yaml`
    overwrite_existing : bool, optional
        If true, will rerun reconstructions if they are found in the output,
        otherwise it will skip
    override_params : dict, optional
        Parameters in the config at `config_path` to override, and their override values.
        Parameters, if nested, should be separated by `.` for each nesting level,
        e.g. {"param.subparam.subsubparam": "override_value"}

    Returns:
    --------
    list[str]
        paths to the saved `.npy` reconstructions
    """
    # do a couple sanity checks on the completeness of the config
    config = helper.read_config(config_path)
    config = _override_config_parameters(config, override_params)
    assert os.path.exists(config.get("base_data_path")), "Must specify data path"
    assert not config.get("passthrough"), "Forward model must not be passthrough"
    assert config.get("save_recon_path"), "Must provide save path"
    assert config.get("num_workers"), "Must provide number of workers for dataloading"
    assert config.get("batch_size"), "Must provide batch size for processing"

    # initialize the model and data
    model = get_model(config, device=config["device"])
    try:
        test_dataloader = _get_preprocessed_dataset_dataloader(
            config["base_data_path"],
            batch_size=config["batch_size"],
            workers=config["num_workers"],
        )
    except Exception as e:
        print("Dataloading failed: ", str(e))
        return []

    _display_config_cli(config)

    results = reconstruct_samples(
        model=model,
        test_dataloader=test_dataloader,
        output_dir=config["save_recon_path"],
        model_name=_get_unique_model_name(config),
        is_fista=config["recon_model_params"]["model_name"] == "fista",
        overwrite=overwrite_existing,
    )
    valid_results = [r for r in results if r is not None]
    print(f"Finished: {len(valid_results)}/{len(results)} reconstructions.")
    return valid_results


def reconstruct_samples(
    model: SSLSimulationModel,
    test_dataloader: DataLoader,
    output_dir: str,
    model_name: str,
    is_fista: bool,
    overwrite: bool = False,
) -> list[Optional[str]]:
    """
    Given a non-passthrough SSLSimulationModel (forward and backward), use the model to run simulated reconstructions
    for every sample in the dataset.

    Resulting reconstructions are saved according to the config file's `save_recon_path` parameter,
    unless overridden by `overide_save_path`.

    Parameters
    ----------
    model: SSLSimulationModel
        Simulation model with non-passthrough forward model
    test_dataloader: Dataloader
        Pytorch dataloader for the test dataset
    output_dir : str
        Directory to save each reconstruction output to.
    model_name : str
        Unique identifier of the model to append to the reconstruction name
    is_fista : bool
        Whether the reconstruction model is fista. Fudge to fix some of the standardization issues
        TODO: Fix fm configuration this so that these run more standardized
    overwrite: bool, optional
        Whether to overwrite existing results. If false, will skip these samples

    Returns
    -------
    list[str]
        List of paths to the saved reconstruction files. If a reconstruction failed, the corresponding entry will be None.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, sample in tqdm.tqdm(
        enumerate(test_dataloader), desc="Running reconstructions"
    ):
        # Batching means that we have a list of paths instead of a single path
        sample_names = [
            os.path.splitext(os.path.basename(p))[0] for p in sample["path"]
        ]
        output_paths = [
            os.path.join(output_dir, "_".join([model_name, name]) + ".npy")
            for name in sample_names
        ]

        if not overwrite and all(os.path.exists(p) for p in output_paths):
            print(f"All batch {i} samples computed, skipping...")
            continue
        with torch.no_grad():
            try:
                ground_truth = sample["image"]

                sim_meas = model.model1(
                    ground_truth.unsqueeze(1).to(model.model1.device)
                )

                # TODO fix this
                # The learned models need simulated measurements to be 0 mean 1 std
                # normalized before reconstruction, while FISTA needs the feature-channel
                # to be removed. This is an artifact of legacy code, and should be fixed
                if is_fista:
                    sim_meas = sim_meas.squeeze(2)
                else:
                    sim_meas = (sim_meas - sim_meas.mean()) / sim_meas.std()

                reconstruction = model.model2(sim_meas).detach().cpu().numpy()

                # TODO fix: fista recons come out (b, x, y, c) instead of (b, c, y, x)
                if is_fista:
                    reconstruction = reconstruction.transpose(0, 3, 1, 2)

                # save off each measurement in the batch
                for sample_output_path, recon in zip(output_paths, reconstruction):
                    np.save(sample_output_path, recon)
                    print(f"\n\tSaved recon {recon.shape} to {sample_output_path}\n")
                    wandb.log({"sample": i, "shape": recon.shape})
                    results.append(sample_output_path)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append([None] * len(output_paths))
                continue
    return results


def compute_metrics(
    config_path: str, overwrite_existing: bool = True, override_params={}, mp_workers=4
) -> Optional[str]:
    """
    Given the path to a config file for which an experiment has already been run, compute
    a set of FRIQ metrics for every reconstruction & ground-truth pair generated
    by this config's experiment.

    Metrics are saved in JSON-valued dictionaries at:
        - <save_recon_path>/<unique-model-name>_metrics.json

    NOTE: reconstruction & ground-truth pairs are matched by the their file names, which
    are expected to be, respectively:
        - <save_recon_path>/<unique-model-name>_<sample-name>.npy
        - <base_data_path>/<sample-name>.mat

    If pairs are NOT found (or are missing), the metrics will skip this sample.

    Parameters
    ----------
    config_path : str
        Path to the config `yaml`
    overwrite_existing : bool, optional
        If true, will rerun reconstructions if they are found in the output,
        otherwise it will skip
    override_params : dict, optional
        Parameters in the config at `config_path` to override, and their override values.
        Parameters, if nested, should be separated by `.` for each nesting level,
        e.g. {"param.subparam.subsubparam": "override_value"}
    mp_workers : int
        Number of workers to allocate for multiprocessed metric computation

    Returns
    -------
    Optional[str]
        Path to the file where metrics are saved, or None if no metrics were written

    """
    config = helper.read_config(config_path)
    config = _override_config_parameters(config, override_params)
    model_name = _get_unique_model_name(config)

    # Determine file paths
    base_data_path = config["base_data_path"]
    save_recon_path = config["save_recon_path"]
    metrics_out_path = os.path.join(save_recon_path, f"{model_name}_metrics.json")

    if os.path.exists(metrics_out_path) and not overwrite_existing:
        print(f"Metrics already computed at {metrics_out_path}, skipping...")
        return

    # Collect matching prediction and ground truth file pairs
    pred_files = sorted(glob.glob(os.path.join(save_recon_path, f"{model_name}_*.npy")))
    gt_files = sorted(glob.glob(os.path.join(base_data_path, "*.mat")))

    gt_sample_names = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
    matched = [
        (pred_path, gt_sample_names.get(_extract_sample_name(pred_path, model_name)))
        for pred_path in pred_files
    ]
    matched = [
        (pred, gt) for pred, gt in matched if gt is not None and os.path.exists(gt)
    ]

    print(f"Found {len(matched)} matched predictions and ground truth samples.")

    results = _calculate_metrics_multiprocessed(matched, mp_workers)
    if not results:
        print("No valid results to aggregate. Skipping JSON write.")
        return

    # Save aggregate and overwrite existing metrics if we've gotten this far
    agg_metrics = {
        m: float(np.mean([r[m] for r in results])) for m in results[0].keys()
    }
    _display_metrics_cli(agg_metrics)
    print(f"Saving metrics to {metrics_out_path}.\n")
    with open(metrics_out_path, "w") as f:
        json.dump(agg_metrics, f, indent=4)

    return metrics_out_path


# ------- Helper functions -------- #


def _get_preprocessed_dataset_dataloader(dataset_path: str, batch_size=1, workers=1):
    """
    This helper assumses that the data has already been preprocessed (properly formatted), and
    lives as the only files in the specified directory as `.mat` files with the "image" tag.

    See the studies readme for more info on how to generate this dataset.
    """
    preprocessed_dataset_paths = glob.glob(os.path.join(dataset_path, "*.mat"))
    dataset = ds.SpectralDataset(
        preprocessed_dataset_paths,
        transform=transforms.Compose(
            [
                ds.Normalize(),  # 0-1 normalize before forward model, as standard
                ds.toTensor(yxc_to_cyx=False),  # dont transpose preprocessed
            ]
        ),
        tag="image",
    )
    return DataLoader(dataset, batch_size, shuffle=False, num_workers=workers)


def _get_unique_model_name(config: dict) -> str:
    """
    Get a unique name for the model described by this config using a selection of parameters.

    TODO: add this as a __name__ for each model instead
    """

    # Format to avoid scientific notation, remove trailing zeros
    def format_value(val):
        if isinstance(val, float):
            return f"{val:.8f}".rstrip("0").rstrip(".")
        return str(val)

    fwd_params = config["forward_model_params"]
    operations = fwd_params["operations"]

    recon_model_name = config["recon_model_params"]["model_name"]
    depth = fwd_params["stack_depth"]
    psf_stride = fwd_params["psf"]["stride"]

    masknoise = (
        fwd_params["mask_noise"]["intensity"] if operations["fwd_mask_noise"] else False
    )
    shotnoise = (
        fwd_params["sample_noise"]["photon_count"]
        if operations["shot_noise"]
        else False
    )
    readnoise = (
        fwd_params["sample_noise"]["intensity"] if operations["read_noise"] else False
    )

    return "-".join(
        [
            f"name={recon_model_name}",
            f"numpsfs={depth}",
            f"psfstride={format_value(psf_stride)}",
            f"masknoise={format_value(masknoise)}",
            f"shotnoise={format_value(shotnoise)}",
            f"readnoise={format_value(readnoise)}",
        ]
    )


def _calculate_metrics_multiprocessed(
    pred_gt_pairs: list[tuple[str, str]], workers: int
) -> list[dict[str, float]]:
    """
    Calculate metrics between a set of reconstruction & ground truth pairs, as a list of
    tuples of filepaths.

    Parameters
    ----------
    pred_gt_pairs : str
        Paths to the reconstruction .npy and GT .mat files
    workers : int
        number of workers to run multiprocessing with

    Returns
    -------
    list[dict[str, float]]
        Metrics for each sample
    """
    args = [(i, pred, gt) for i, (pred, gt) in enumerate(pred_gt_pairs)]
    with mp.Pool(processes=workers) as pool:
        total = len(args)
        jobs = [
            pool.apply_async(_calculate_metrics, args=(total, idx, pred, gt))
            for idx, pred, gt in args
        ]
        results = [job.get() for job in jobs if job.get() is not None]
    return results


def _calculate_metrics(
    total: int, idx: int, pred_file, gt_file
) -> Optional[dict[str, float]]:
    """
    Helper function for a single sample's calculate metrics job. Returns a
    JSON dictionary of calculated metrics for the provided sample pair.

    If processing fails will retun None,
    """
    sample_scores = {}

    try:
        # TODO standardize this dataloading format
        pred = np.load(pred_file)
        gt = io.loadmat(gt_file)["image"]

        if pred.shape[0] == 32:
            pred = pred[1:-1]

        # standardize the intensity across all samples for uniform comparison
        pred = (pred - np.mean(pred)) / np.std(pred)
        gt = (gt - np.mean(gt)) / np.std(gt)

        if np.any(np.isnan(pred)) or np.any(np.isnan(gt)):
            print(f"NaN in {os.path.basename(pred_file)}... Skipping.")
            return None

        for metric in METRICS:
            score = metrics.get_score(metric, pred, gt)
            if np.isnan(score):
                print(f"NaN {metric} in {os.path.basename(pred_file)}... Skipping.")
                return None
            sample_scores[metric] = score

        print(f"Processed {idx+1} / {total}", end="\r")
        return sample_scores

    except Exception as e:
        print(f"Error processing {os.path.basename(pred_file)}: {e}")
        return None


def _extract_sample_name(pred_path: str, model_prefix: str) -> str:
    """
    Get the stem basename of the sample from the predicted path to enable
    proper sample-gt pairing.

    Assumes format for the prediction path:
        - <unique-model-name>_<sample-name>.npy
    """
    filename = os.path.basename(pred_path)
    prefix = f"{model_prefix}_"
    if filename.startswith(prefix):
        sample_name = filename[len(prefix) :].replace(".npy", "")
    else:
        sample_name = filename.replace(".npy", "")
    return sample_name


def _override_config_parameters(config: dict, overrides: dict[str, Any]) -> dict:
    """
    Override the parameters in a configuration dictionary with specified values.

    Parameters in the overrides should be keyed `parameter_name` : `new_value`.
    If nested,`parameter_name` should be a string separated by `.` for each
    nesting level, e.g. {"param.subparam.subsubparam": "override_value"}
    """
    for parameter_name, value in overrides.items():
        param_levels = parameter_name.split(".")
        d = config
        for key in param_levels[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[param_levels[-1]] = value
    return config


def _display_config_cli(config: dict):
    """Helper for a cli printout of the config file"""
    config_name = os.path.basename(config["config_fname"])
    chars = len(config_name)
    print("\n")
    print("#" * (38 + chars))
    print("#" * 10, f" Loaded config {config_name} ", "#" * 10)
    print("#" * (38 + chars))
    print("\n")
    pprint(config)
    print("\n")
    print("#" * (39 + chars))
    print("#" * 10, f" Running config {config_name} ", "#" * 10)
    print("#" * (39 + chars))
    print("\n")


def _display_metrics_cli(metrics: dict):
    """Helper for a CLI printout of the computed metrics"""
    print("\n")
    print("#" * 48)
    print("#" * 15 + " Computed Metrics " + "#" * 15)
    print("#" * 48)
    print("\n")
    pprint(metrics)
    print("#" * 48)
