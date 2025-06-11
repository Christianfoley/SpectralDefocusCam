# %%
import json
import re
import glob
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

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


def select_indices_for_noise_pairs(json_paths, read_and_shot_noise_pairs):
    """
    Select indices of json_paths matching any (readnoise, shotnoise) pair.

    Parameters
    ----------
    json_paths : list of str
        List of file paths.
    read_and_shot_noise_pairs : iterable of (float, float)
        (readnoise, shotnoise) pairs to match.

    Returns
    -------
    list of int
        Indices of json_paths that match any of the noise pairs.
    """
    # Convert to set for O(1) lookup
    target_pairs = set(read_and_shot_noise_pairs)

    indices = []
    for idx, path in enumerate(json_paths):
        readnoise, shotnoise = extract_noise_from_path(path)
        if readnoise and shotnoise:
            if (readnoise, shotnoise) in target_pairs:
                indices.append(idx)
    return indices


def plot_metrics_dual_x(
    metric_json_paths_list,
    model_names,
    color_indices,
    metric_name="mse",
):
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    colors = sns.husl_palette(n_colors=10, l=0.5)

    plt.figure(figsize=(10, 6), dpi=200)
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

        sorted_items = sorted(zip(photon_counts, read_noises, scores))
        photon_counts, read_noises, scores = map(list, zip(*sorted_items))

        all_photon_counts.extend(photon_counts)
        all_read_noises.extend(read_noises)

        ax.plot(
            photon_counts,
            scores,
            marker="o",
            color=colors[color_indices[model_idx]],
            label=model_names[model_idx],
            linewidth=3,
            markersize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Photon Count (Shot Noise)", fontsize=14)
    ax.set_ylabel(metric_name.upper(), fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # LogLocator for log axis ticks
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Dual top axis: map photon count to inverse-mapped read noise
    sorted_unique = sorted(set(zip(all_photon_counts, all_read_noises)))
    photons_sorted, reads_sorted = zip(*sorted_unique)

    log_photons = np.log10(photons_sorted)
    log_reads = np.log10(reads_sorted)

    def readnoise_forward(x):
        logx = np.log10(x)
        return 10 ** np.interp(logx, log_photons, log_reads[::-1])

    def readnoise_inverse(x):
        logx = np.log10(x)
        return 10 ** np.interp(logx, log_reads[::-1], log_photons)

    secax = ax.secondary_xaxis("top", functions=(readnoise_forward, readnoise_inverse))
    secax.set_xscale("log")
    secax.set_xlabel("Read Noise", fontsize=14)
    secax.tick_params(labelsize=12)
    secax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    secax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))

    ax.legend(fontsize=13)
    plt.title(f"Metric '{metric_name.upper()}' vs Noise Levels", fontsize=16)
    plt.tight_layout()
    plt.show()


# %%
if __name__ == "__main__":
    # TODO wait for FISTA results
    metrics_files = glob.glob(
        "/home/cfoley/SpectralDefocusCam/studies/simulation_noise_tolerance_ablation/outputs/defocuscam_learned/*.json"
    )

    noise_pairs = list(
        zip(
            [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1],
            [50, 100, 500, 1000, 10000, 100000],
        )
    )
    selected_metrics_files = [
        f
        for i, f in enumerate(metrics_files)
        if i in select_indices_for_noise_pairs(metrics_files, noise_pairs)
    ]

    plot_metrics_dual_x([selected_metrics_files], ["learned"], [0], metric_name="psnr")
    plot_metrics_dual_x([selected_metrics_files], ["learned"], [0], metric_name="mse")
    plot_metrics_dual_x([selected_metrics_files], ["learned"], [0], metric_name="ssim")
    plot_metrics_dual_x(
        [selected_metrics_files], ["learned"], [0], metric_name="cossim"
    )
# %%
