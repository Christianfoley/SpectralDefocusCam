import os
import pathlib
import re
import glob

import seaborn as sns

from defocuscam.studies.simulation_noise_tolerance_ablation.run import (
    SHOT_NOISE_PHOTON_COUNT,
    READ_NOISE_INTENSITY,
    CALIBRATION_NOISE_INTENSITY,
    generate_sweep_configs,
)
from defocuscam.studies.simulation_numbers_of_defocus_ablation.plot import (
    generate_metrics_plots_for_one_ablation,
)


sns.set_theme(style="whitegrid", font_scale=1.2)


def extract_noise_from_path(path):
    """Extract sigmas and photon count from the filename."""
    match = re.search(r"masknoise=([0-9e.-]+)-shotnoise=([0-9e.-]+)-readnoise=([0-9e.-]+)", path)
    if match:
        masknoise = float(match.group(1))
        shotnoise = float(match.group(2))
        readnoise = float(match.group(3))
        return masknoise, readnoise, shotnoise
    return None, None, None


def main():
    METRICS_TO_PLOT = ["mse", "psnr", "ssim", "cossim"]
    METRIC_YLIMS = {
        "mse": (-0.1, 1.9),
        "psnr": (-4, 21),
        "ssim": (-0.1, 1.03),
        "cossim": (-0.1, 1.08),
    }
    OUT_PATH = os.path.join(pathlib.Path(__file__).parent, "results")

    configs = glob.glob(os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml"))
    configs, overrides, suffixes = generate_sweep_configs(configs)
    print(f"Collected {len(configs)} configs:")
    for config_path in configs:
        print("\t", os.path.abspath(config_path))
    print("\n")

    # Generate the plots for each ablation separately
    suffix_key = "read"
    generate_metrics_plots_for_one_ablation(
        configs=[cfg for cfg, suffix in zip(configs, suffixes) if suffix_key in suffix],
        overrides=[ovr for ovr, suffix in zip(overrides, suffixes) if suffix_key in suffix],
        ablation_name="read_noise",
        out_path=os.path.join(OUT_PATH, suffix_key + "_noise"),
        expected_datapoints=len(READ_NOISE_INTENSITY),
        metric_names=METRICS_TO_PLOT,
        logscale_x=True,
        ablation_x_patterns=[r"readnoise=([0-9.]+|[Ff]alse)"] * 2,
        ablation_x_name="Read Noise (sigma)",
        plot_size=(8, 5),
        flip_x=False,
        ylims=METRIC_YLIMS,
        linestyles=["-"] * len(configs),
    )

    suffix_key = "shot"
    generate_metrics_plots_for_one_ablation(
        configs=[cfg for cfg, suffix in zip(configs, suffixes) if suffix_key in suffix],
        overrides=[ovr for ovr, suffix in zip(overrides, suffixes) if suffix_key in suffix],
        ablation_name="shot_noise",
        out_path=os.path.join(OUT_PATH, suffix_key + "_noise"),
        expected_datapoints=len(SHOT_NOISE_PHOTON_COUNT),
        metric_names=METRICS_TO_PLOT,
        logscale_x=True,
        ablation_x_patterns=[r"shotnoise=([0-9.]+|[Ff]alse)"] * 2,
        ablation_x_name="Shot Noise (photons)",
        plot_size=(8, 5),
        flip_x=True,
        ylims=METRIC_YLIMS,
        linestyles=["-"] * len(configs),
    )
    suffix_key = "mask"
    generate_metrics_plots_for_one_ablation(
        configs=[cfg for cfg, suffix in zip(configs, suffixes) if suffix_key in suffix],
        overrides=[ovr for ovr, suffix in zip(overrides, suffixes) if suffix_key in suffix],
        ablation_name="mask_noise",
        out_path=os.path.join(OUT_PATH, suffix_key + "_noise"),
        expected_datapoints=len(CALIBRATION_NOISE_INTENSITY),
        metric_names=METRICS_TO_PLOT,
        logscale_x=True,
        ablation_x_patterns=[r"masknoise=([0-9.]+|[Ff]alse)"] * 2,
        ablation_x_name="Calibration Noise (sigma)",
        plot_size=(8, 5),
        flip_x=False,
        ylims=METRIC_YLIMS,
        linestyles=["-"] * len(configs),
    )


if __name__ == "__main__":
    main()
