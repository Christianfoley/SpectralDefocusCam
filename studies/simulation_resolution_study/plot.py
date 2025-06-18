import glob
from cleanplots import *  # noqa: F403
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from typing import Literal
import os
import pathlib

import utils.helper_functions as helper
from studies.run_study_utils import _get_unique_model_name

sns.set_context("notebook", font_scale=1.2)
COLORS = sns.husl_palette(n_colors=10, l=0.5)
NOISY_COLOR = COLORS[0]
NORMAL_COLOR = COLORS[7]


def load_and_process_recon(recon_path: str, crop_coords: tuple) -> np.ndarray:
    """Load and process reconstruction data from .npy file.

    Parameters
    ----------
    recon_path : str
        Path to reconstruction .npy file, in CYX format
    crop_coords : tuple
        Tuple of (y_start, y_end, x_start, x_end) for cropping

    Returns
    -------
    np.ndarray
        Processed reconstruction data
    """
    y1, y2, x1, x2 = crop_coords
    recon = np.load(recon_path).transpose(1, 2, 0).squeeze()[y1:y2, x1:x2, :]
    return recon


def plot_single_image_with_line(img, direction, color, linestyle):
    fig = plt.figure(figsize=(5, 5), dpi=500)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])  # Create axes with no padding
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap="gray")

    if direction == "horizontal":
        ax.plot([0, 12], [6, 6], color=color, linestyle=linestyle, linewidth=4)
    else:
        ax.plot([6, 6], [0, 12], color=color, linestyle=linestyle, linewidth=4)

    return fig


def plot_spatial_profile_and_recon(
    recon_normal: np.ndarray,
    recon_noisy: np.ndarray,
    direction: Literal["horizontal", "vertical"],
    peak_height: float = 0.5,
    threshold_ratio: float = 0.7,
    recon_name: str = "",
    out_path: str = "",
) -> tuple[plt.Figure, plt.Figure]:
    """Plot spatial profile and peaks for normal and noisy reconstructions.

    Parameters
    ----------
    recon_normal : np.ndarray
        Normal reconstruction data
    recon_noisy : np.ndarray
        Noisy reconstruction data
    """
    if out_path:
        out_path_dir = os.path.dirname(out_path)
        os.makedirs(out_path_dir, exist_ok=True)
        out_name_suffix = pathlib.Path(out_path).stem

    # Plot mean images side by side
    mean_img_normal = np.mean(recon_normal, 2)
    mean_img_noisy = np.mean(recon_noisy, 2)
    if direction == "horizontal":
        profile_normal = mean_img_normal[6, :]
        profile_noisy = mean_img_noisy[6, :]
    else:
        profile_normal = mean_img_normal[:, 6]  # noqa: F841
        profile_noisy = mean_img_noisy[:, 6]

    # Skipping for plots in the paper as so similar it obscures the plot
    # fig1 = plot_single_image_with_line(mean_img_normal, direction, NORMAL_COLOR, "--")
    # if out_path:
    #     with open(
    #         os.path.join(out_path_dir, out_name_suffix + "_cropped_recon.png"), "wb"
    #     ) as f:
    #         plt.savefig(f)
    # plt.close(fig1)

    fig2 = plot_single_image_with_line(mean_img_noisy, direction, NOISY_COLOR, "--")
    if out_path:
        with open(
            os.path.join(out_path_dir, out_name_suffix + "_cropped_recon_noisy.png"),
            "wb",
        ) as f:
            plt.savefig(f)
    plt.close(fig2)

    # Plot profiles with different line styles
    fig3 = plt.figure(dpi=500, figsize=(5, 5))

    # Skipping for plots in the paper as so similar it obscures the plot
    # # Normal profile
    # profile_normal = profile_normal / np.max(profile_normal)
    # peaks_normal, _ = find_peaks(profile_normal, height=peak_height)
    # plt.plot(
    #     profile_normal,
    #     color=NORMAL_COLOR,
    #     linestyle="--",
    #     alpha=1.0,
    #     linewidth=3,
    #     label="Normal",
    # )
    # if len(peaks_normal) > 1:
    #     plt.axhline(
    #         y=profile_normal[peaks_normal[1]] * threshold_ratio,
    #         color=NORMAL_COLOR,
    #         linestyle="--",
    #         alpha=0.7,
    #     )

    # Noisy profile
    profile_noisy = profile_noisy / np.max(profile_noisy)
    peaks_noisy, _ = find_peaks(profile_noisy, height=peak_height)
    plt.plot(
        profile_noisy,
        color=NOISY_COLOR,
        linestyle="-",
        alpha=1.0,
        linewidth=3,
        label="Noisy",
    )
    # if len(peaks_noisy) > 1:
    #     plt.axhline(
    #         y=profile_noisy[peaks_noisy[1]] * threshold_ratio,
    #         color=NOISY_COLOR,
    #         linestyle=":",
    #         alpha=0.7,
    #     )

    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(-0.15, 1.25))
    if recon_name:
        plt.title(recon_name)

    if out_path:
        with open(
            os.path.join(out_path_dir, out_name_suffix + "_spatial_profile.png"), "wb"
        ) as f:
            plt.savefig(f)

    plt.close(fig3)
    return fig2, fig3


def plot_spectral_profile(
    recon_path_normal: str,
    recon_path_noisy: str,
    filter_path: str,
    peak_height: float = 0.5,
    recon_name: str = "",
    threshold_ratio: float = 0.7,
    out_path: str = "",
) -> plt.Figure:
    """Plot spectral profile with wavelength axis for normal and noisy reconstructions.

    Parameters
    ----------
    recon_path_normal : str
        Path to normal reconstruction file
    recon_path_noisy : str
        Path to noisy reconstruction file
    ...existing parameters...
    """
    if out_path:
        out_path_dir = os.path.dirname(out_path)
        os.makedirs(out_path_dir, exist_ok=True)
        out_name_suffix = pathlib.Path(out_path).stem

    # Load filter data
    filter_data = scipy.io.loadmat(filter_path)
    wavelengths = filter_data["waves"].squeeze()

    # Load and normalize reconstructions
    recon_normal = np.load(recon_path_normal).transpose(1, 2, 0).squeeze()[127, 127, :]
    recon_normal = recon_normal / recon_normal.max()

    recon_noisy = np.load(recon_path_noisy).transpose(1, 2, 0).squeeze()[127, 127, :]
    recon_noisy = recon_noisy / recon_noisy.max()

    plt.rcParams["font.sans-serif"] = "Arial"
    fig = plt.figure(dpi=500, figsize=(5, 5))

    # Skipping for plots in the paper as so similar it obscures the plot
    # # Plot normal profile
    # plt.plot(
    #     wavelengths,
    #     recon_normal,
    #     color=NORMAL_COLOR,
    #     linestyle="--",
    #     alpha=1.0,
    #     linewidth=3,
    #     label="Normal",
    # )
    # peaks_normal, _ = find_peaks(recon_normal, height=peak_height)
    # if len(peaks_normal) > 0:
    #     peak_idx = min(1, len(peaks_normal) - 1)
    #     plt.axhline(
    #         y=recon_normal[peaks_normal[peak_idx]] * threshold_ratio,
    #         color=NORMAL_COLOR,
    #         linestyle="--",
    #         alpha=0.7,
    #     )

    # Plot noisy profile
    plt.plot(
        wavelengths,
        recon_noisy,
        color=NOISY_COLOR,
        linestyle="-",
        alpha=1,
        linewidth=3,
        label="Noisy",
    )
    peaks_noisy, _ = find_peaks(recon_noisy, height=peak_height)
    # if len(peaks_noisy) > 0:
    #     peak_idx = min(1, len(peaks_noisy) - 1)
    #     plt.axhline(
    #         y=recon_noisy[peaks_noisy[peak_idx]] * threshold_ratio,
    #         color=NOISY_COLOR,
    #         linestyle=":",
    #         alpha=0.7,
    #     )

    plt.yticks([])
    plt.xticks([450, 525, 600, 675, 750, 825], fontsize=24)
    plt.xlabel("Wavelength (nm)", fontsize=24)
    plt.legend(bbox_to_anchor=(-0.15, 1.25))
    if recon_name:
        plt.title(recon_name)

    if out_path:
        with open(
            os.path.join(out_path_dir, out_name_suffix + "_spectral_profile.png"), "wb"
        ) as f:
            plt.savefig(f)
    plt.close(fig)
    return fig


def main():
    CROP_COORDS = {"vertical": (122, 135, 121, 134), "horizontal": (121, 134, 122, 135)}
    OUT_DIR = os.path.join(os.path.join(pathlib.Path(__file__).parent, "results"))

    # Get base configs (non-noisy versions)
    base_configs = [
        c
        for c in glob.glob(
            os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml")
        )
        if "_noisy_" not in c
    ]

    print(f"Collected {len(base_configs)} base configs:")
    for config_path in base_configs:
        print("\t", os.path.abspath(config_path))
    print("\n")

    for base_config_path in base_configs:
        # Construct noisy config path by inserting "_noisy" before the type
        base_name = os.path.basename(base_config_path)
        type_start = (
            base_name.find("_spatial")
            if "_spatial" in base_name
            else base_name.find("_spectral")
        )
        noisy_config_path = base_config_path.replace(
            base_name, f"{base_name[:type_start]}_noisy{base_name[type_start:]}"
        )

        if not os.path.exists(noisy_config_path):
            print(f"Warning: No matching noisy config found for {base_config_path}")
            continue

        base_config = helper.read_config(base_config_path)
        noisy_config = helper.read_config(noisy_config_path)

        # Use base config name for output directory
        base_model_name = _get_unique_model_name(base_config)
        noisy_model_name = _get_unique_model_name(noisy_config)

        gt_filenames = sorted(os.listdir(base_config["base_data_path"]))

        for gt_filename in gt_filenames:
            recon_name = os.path.splitext(gt_filename)[0]

            base_recon_path = os.path.join(
                base_config["save_recon_path"],
                "_".join([base_model_name, recon_name + ".npy"]),
            )
            noisy_recon_path = os.path.join(
                noisy_config["save_recon_path"],
                "_".join([noisy_model_name, recon_name + ".npy"]),
            )

            # Use base config name for output path, making sure to specify direction
            is_spatial = "spatial" in base_config["base_data_path"]
            peak_height = 0.25 if "wavelength_714" in base_recon_path else 0.5
            direction = (
                "vertical"
                if "vertical" in base_config["base_data_path"]
                else "horizontal"
            )

            if is_spatial:
                recon_name = "_".join([recon_name, direction])
                base_recon = load_and_process_recon(
                    base_recon_path, CROP_COORDS[direction]
                )
                noisy_recon = load_and_process_recon(
                    noisy_recon_path, CROP_COORDS[direction]
                )

                recon_crop, spatial_profile = plot_spatial_profile_and_recon(
                    base_recon,
                    noisy_recon,
                    direction,
                    recon_name=recon_name,
                    out_path=os.path.join(OUT_DIR, recon_name),
                )
                recon_crop.show()
                spatial_profile.show()
            else:
                spectral_profile = plot_spectral_profile(
                    base_recon_path,
                    noisy_recon_path,
                    base_config["mask_dir"],
                    peak_height=peak_height,
                    recon_name=recon_name,
                    out_path=os.path.join(OUT_DIR, recon_name),
                )
                spectral_profile.show()


if __name__ == "__main__":
    main()
