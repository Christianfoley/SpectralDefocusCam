import collections
import json
import os
import pathlib
import re
import glob
from typing import Literal, Optional
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator

import utils.helper_functions as helper
from studies.run_study_utils import _get_unique_model_name, _override_config_parameters

sns.set_theme(style="whitegrid", font_scale=1.2)


def extract_noise_from_path(path):
    """Extract sigma and photon count from the filename."""
    match = re.search(r"shotnoise=([0-9e.-]+)-readnoise=([0-9e.-]+)", path)
    if match:
        shotnoise = float(match.group(1))
        readnoise = float(match.group(2))
        return readnoise, shotnoise
    return None, None


def load_metrics_from_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_metrics_dual_x(
    metric_json_paths_list: list[list[str]],
    model_names: list[str],
    out_base_path: str,
    plot_savename: str = "plot_metrics",
    metric_name: Literal["mse", "psnr", "ssim", "cossim"] = "mse",
    color_indices: Optional[list[int]] = None,
):
    """
    Plots a metric (e.g., MSE) against two noise axes for multiple models:
        - sigma (read noise) [primary x-axis],
        - photon count (shot noise) [secondary top axis]

    Saves plot to <out_base_path>/<plot_name>_<metric_name>.png

    Parameters
    ----------
    metric_json_paths_list : list[list[str]]
        A list where each element is a list of JSON file paths for a model. Each JSON file should
        contain metric results and encode noise parameters in its path.
    model_names : list[str]
        List of model names corresponding to each set of JSON paths.
    out_base_path : str,
        Directory to save the file to
    plot_savename : str
        Base name of the plot to save to
    metric_name : Literal['mse', 'psnr', 'ssim', 'cossim'], optional
        Name of the metric to plot (default is "mse").
    color_indices : Optional[list[int]], optional
        List of indices to select colors for each model from the seaborn palette. If empty will be
        automatically created.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    colors = sns.husl_palette(n_colors=10, l=0.5)
    color_indices = color_indices or [(i * 7) % 10 for i in range(10)]

    fig = plt.figure(figsize=(10, 6), dpi=200)
    ax = plt.gca()

    all_photon_counts = []
    all_read_noises = []

    for model_idx, json_paths in enumerate(metric_json_paths_list):
        read_noises = []
        photon_counts = []
        scores = []

        for path in json_paths:
            readn, photon = extract_noise_from_path(path)
            if readn is None or photon is None:
                continue
            read_noises.append(readn)
            photon_counts.append(photon)
            metrics = load_metrics_from_json(path)
            scores.append(metrics.get(metric_name, None))

        sorted_items = sorted(zip(read_noises, photon_counts, scores))
        read_noises, photon_counts, scores = map(list, zip(*sorted_items))

        all_read_noises.extend(read_noises)
        all_photon_counts.extend(photon_counts)

        print(f"Plotting {metric_name} scores for {model_names[model_idx]}: ")
        print(f"\t{scores}")

        ax.plot(
            read_noises,
            scores,
            marker="o",
            color=colors[color_indices[model_idx]],
            label=model_names[model_idx],
            linewidth=6,
            markersize=18,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Read Noise", fontsize=14)
    ax.set_ylabel(metric_name.upper(), fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, which="both", linestyle="solid", linewidth=1.5)

    # Dual top axis: map read noise to photon count
    sorted_unique = sorted(set(zip(all_read_noises, all_photon_counts)))
    reads_sorted, photons_sorted = zip(*sorted_unique)

    log_reads = np.log10(reads_sorted)
    log_photons = np.log10(photons_sorted)

    def photon_forward(x):
        logx = np.log10(x)
        return 10 ** np.interp(logx, log_reads, log_photons[::-1])

    def photon_inverse(x):
        logx = np.log10(x)
        return 10 ** np.interp(logx, log_photons[::-1], log_reads)

    secax = ax.secondary_xaxis("top", functions=(photon_forward, photon_inverse))
    secax.set_xscale("log")
    secax.set_xlabel("Photon Count (Shot Noise)", fontsize=14)
    secax.tick_params(labelsize=12)
    secax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    secax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))

    ax.legend(
        fontsize=14,
        loc="center left",
        bbox_to_anchor=(-0.15, 1.25),
        borderaxespad=0,
    )
    plt.tight_layout()

    out_path = os.path.join(
        out_base_path, "_".join([plot_savename, metric_name]) + ".png"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"\nSaved plot to:\n\t{out_path}\n")

    return fig


def main():
    # Pairs to plot
    NOISE_PAIRS_TO_PLOT = [
        (0.000001, 10),
        (0.000005, 50),
        (0.00001, 100),
        (0.00005, 500),
        (0.0001, 1000),
        (0.001, 10000),
        (0.01, 100000),
        (0.05, 500000),
        (0.1, 1000000),
    ]
    METRICS_TO_PLOT = ["mse", "psnr", "ssim", "cossim"]
    OUT_PATH = os.path.join(pathlib.Path(__file__).parent, "results")

    configs = glob.glob(os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml"))
    print(f"Collected {len(configs)} configs:")
    for config_path in configs:
        print("\t", os.path.abspath(config_path))
    print("\n")

    metrics_files = collections.defaultdict(lambda: [])
    for config_path in configs:
        config = helper.read_config(config_path)
        model_name = config["recon_model_params"]["model_name"]

        for sigma, photon_count in NOISE_PAIRS_TO_PLOT:
            config = _override_config_parameters(
                config,
                overrides={
                    "forward_model_params.mask_noise.intensity": float(sigma),
                    "forward_model_params.sample_noise.intensity": float(sigma),
                    "forward_model_params.sample_noise.photon_count": float(
                        photon_count
                    ),
                },
            )

            all_metrics_files = glob.glob(
                os.path.join(config["save_recon_path"], "*.json")
            )
            metrics_files[model_name] += [
                fp for fp in all_metrics_files if _get_unique_model_name(config) in fp
            ]
        assert len(metrics_files[model_name]) == len(
            NOISE_PAIRS_TO_PLOT
        ), "Unable to find metrics for all noise pairs. Are all metrics files correctly named?"

    for metric_name in METRICS_TO_PLOT:
        plot_metrics_dual_x(
            metric_json_paths_list=[*metrics_files.values()],
            model_names=[*metrics_files.keys()],
            out_base_path=OUT_PATH,
            metric_name=metric_name,
        )


if __name__ == "__main__":
    main()
# %%
