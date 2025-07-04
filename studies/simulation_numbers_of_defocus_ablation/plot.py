import collections
import json
import os
import pathlib
import re
import glob
from typing import Literal, Optional

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator

import utils.helper_functions as helper
from studies.run_study_utils import _get_unique_model_name, _override_config_parameters

sns.set_theme(style="whitegrid", font_scale=1.2)


def load_metrics_from_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def collect_config_metrics_jsons(config_path, overrides={}):
    """
    Get the metrics jsons corresponding to this config's experiments.
    """
    config = helper.read_config(config_path)
    config = _override_config_parameters(config, overrides)
    model_name = config["recon_model_params"]["model_name"]

    model_metrics_files = [
        fp
        for fp in glob.glob(os.path.join(config["save_recon_path"], "*.json"))
        if _get_unique_model_name(config) in fp
    ]
    return model_name, model_metrics_files


def plot_metrics(
    metric_json_paths_list: list[list[str]],
    model_names: list[str],
    out_base_path: str,
    plot_savename: str = "plot_metrics",
    metric_name: Literal["mse", "psnr", "ssim", "cossim"] = "mse",
    color_indices: Optional[list[int]] = None,
    x_axis_name: str = "values",
    x_patterns: list[str] = [r"numpsfs=([0-9.]+)"] * 2,
    logscale_x: bool = True,
    x_label: Optional[str] = None,
    plot_size: tuple[int, int] = (10, 6),
    flip_x: bool = False,
    ylim: Optional[tuple[int, int]] = None,
    linestyles: list[str] = ["-", "-", "--", "--", ":"],
):
    """
    Plotting function for metrics vs. any extracted variable (e.g., noise, depth).

    Parameters
    ----------
    metric_json_paths_list : list[list[str]]
        Lists of JSON paths for each model.
    model_names : list[str]
        Names of models corresponding to the above.
    out_base_path : str
        Where to save the plot.
    plot_savename : str
        Output base filename.
    metric_name : Literal["mse", "psnr", "ssim", "cossim"]
        Metric to plot.
    color_indices : Optional[list[int]]
        Indices for color palette.
    x_axis_name : str
        Name of x-axis variable.
    x_patterns : list[str]
        Regex to extract x values from paths for each model
        TODO: update this to take these directly
    logscale_x : bool
        Whether to apply log scale to x-axis.
    x_label : Optional[str]
        Custom x-axis label (default uses `x_axis_name`)
    plot_size : tuple[int, int]
        Size of the plot (width, height) in inches
    flip_x : bool
        Whether to flip the x axis (descending vs ascending)
    ylim : Optional[tuple[int, int]]
        Lower and upper limits for the y axis
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    colors = sns.husl_palette(n_colors=10, l=0.5)
    color_indices = color_indices or [(i * 7) % 10 for i in range(10)]

    fig = plt.figure(figsize=plot_size, dpi=300)
    ax = plt.gca()

    for model_idx, json_paths in enumerate(metric_json_paths_list):
        x_vals = []
        scores = []

        for path in json_paths:
            match = re.search(x_patterns[model_idx], path)
            if not match:
                continue
            elif match.group(1) == "False":
                x_val = 0
            else:
                x_val = float(match.group(1))
            metrics = load_metrics_from_json(path)
            score = metrics.get(metric_name, None)
            if score is None:
                continue
            x_vals.append(x_val)
            scores.append(score)

        if not x_vals:
            continue

        sorted_items = sorted(zip(x_vals, scores))
        x_vals, scores = map(list, zip(*sorted_items))

        print(f"Plotting {metric_name} scores for {model_names[model_idx]}:")
        print(f"\t{scores}")

        ax.plot(
            x_vals,
            scores,
            marker="o",
            color=colors[color_indices[model_idx]],
            label=model_names[model_idx],
            linewidth=6,
            markersize=18,
            linestyle=linestyles[model_idx],
        )

    # Axis formatting
    ax.set_xlabel(x_label or x_axis_name.replace("_", " ").title(), fontsize=14)
    ax.set_ylabel(metric_name.upper(), fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    if logscale_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if flip_x:
        plt.gca().invert_xaxis()

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if ylim is not None:
        ax.set_ylim(ylim)

    # Spine and grid styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.grid(True, which="both", linestyle="solid", linewidth=1.5)

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


def generate_metrics_plots_for_one_ablation(
    configs,
    overrides,
    ablation_name,
    out_path,
    expected_datapoints,
    metric_names,
    logscale_x,
    ablation_x_patterns,
    ablation_x_name,
    plot_size,
    flip_x,
    ylims,
    linestyles,
):
    """Helper subroutine to clean up some boilerplate"""
    # Collect the metrics derived from to those configs experiments
    model_jsons = collections.defaultdict(list)
    for config_path, config_overrides in zip(configs, overrides):
        model_name, metric_jsons = collect_config_metrics_jsons(
            config_path, config_overrides
        )
        model_jsons[model_name] += metric_jsons

    assert all(
        len(files) == expected_datapoints for files in model_jsons.values()
    ), f"Unable to find metrics for all {ablation_name} values. Are all metrics files correctly named?"

    # Plot the metrics
    for metric_name in metric_names:
        plot_metrics(
            metric_json_paths_list=[*model_jsons.values()],
            model_names=[*model_jsons.keys()],
            out_base_path=out_path,
            metric_name=metric_name,
            logscale_x=logscale_x,
            x_patterns=ablation_x_patterns,
            x_axis_name=ablation_x_name,
            plot_size=plot_size,
            flip_x=flip_x,
            ylim=ylims[metric_name] if ylims is not None else ylims,
            linestyles=linestyles,
        )


def main():
    # num psfs and psfs stride
    NUM_PSFS_TO_PLOT = [
        (1, -1),  # blurry only
        (2, 4),  # first and last
        (3, 2),  # 3
        (5, 1),  # all 5
    ]
    METRICS_TO_PLOT = ["mse", "psnr", "ssim", "cossim"]
    OUT_PATH = os.path.join(pathlib.Path(__file__).parent, "results")

    # Collect our configs
    configs = glob.glob(os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml"))
    print(f"Collected {len(configs)} configs:")
    for config_path in configs:
        print("\t", os.path.abspath(config_path))
    print("\n")

    # Generate a plot for each metric
    generate_metrics_plots_for_one_ablation(
        configs=configs,
        overrides=[{}] * len(configs),
        ablation_name="stack_depth",
        out_path=OUT_PATH,
        expected_datapoints=len(NUM_PSFS_TO_PLOT),
        metric_names=METRICS_TO_PLOT,
        logscale_x=False,
        ablation_x_patterns=[r"numpsfs=([0-9.]+)"] * len(configs),
        ablation_x_name="Number of Measurements",
        plot_size=(10, 6),
        flip_x=False,
        ylims=None,  # autoscale all
        linestyles=["-"] * len(configs),
    )


if __name__ == "__main__":
    main()
