import glob
from cleanplots import *
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from typing import Literal
import os
import pathlib

import utils.helper_functions as helper
from studies.run_study_utils import _get_unique_model_name


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


def plot_spatial_profile(
    recon: np.ndarray,
    direction: Literal["horizontal", "vertical"],
    peak_height: float = 0.5,
    threshold_ratio: float = 0.75,
    recon_name: str = "",
    out_path: str = "",
) -> tuple[plt.Figure, plt.Figure]:
    """Plot spatial profile and peaks.

    Parameters
    ----------
    recon : np.ndarray
        Reconstruction data
    direction : Literal["horizontal", "vertical"]
        Profile direction
    peak_height : float
        Minimum height for peak detection
    threshold_ratio : float
        Ratio for horizontal threshold line
    recon_name : str
        Informative name of the reconstruction. If provided will
        be used as plot title
    out_path : str
        Path to output file prefix. If provided, figures will be saved
        with _recon_crop and _spatial_profile suffixes
    """
    if out_path:
        out_path_dir = os.path.dirname(out_path)
        os.makedirs(out_path_dir, exist_ok=True)
        out_name_suffix = pathlib.Path(out_path).stem

    fig1 = plt.figure(dpi=150)
    mean_img = np.mean(recon, 2)
    plt.imshow(mean_img, cmap="gray")
    plt.axis("off")

    if direction == "horizontal":
        plt.plot([0, 12], [6, 6], "r--", linewidth=4)
        profile = mean_img[6, :]
    else:
        plt.plot([6, 6], [0, 12], "r--", linewidth=4)
        profile = mean_img[:, 6]

    if out_path:
        with open(
            os.path.join(out_path_dir, out_name_suffix + "_recon_crop.png"), "wb"
        ) as f:
            plt.savefig(f)

    # Plot profile
    profile = profile / np.max(profile)
    peaks, _ = find_peaks(profile, height=peak_height)

    fig2 = plt.figure(dpi=150, figsize=(5, 5))
    plt.plot(profile, "r")
    plt.xticks([])
    plt.yticks([])
    if len(peaks) > 1:
        plt.axhline(y=profile[peaks[1]] * threshold_ratio, color="r", linestyle="--")
    if recon_name:
        plt.title(recon_name)

    if out_path:
        with open(
            os.path.join(out_path_dir, out_name_suffix + "_spatial_profile.png"), "wb"
        ) as f:
            plt.savefig(f)

    return fig1, fig2


def plot_spectral_profile(
    recon_path: str,
    filter_path: str,
    peak_height: float = 0.5,
    recon_name: str = "",
    threshold_ratio: float = 0.75,
    out_path: str = "",
) -> plt.Figure:
    """Plot spectral profile with wavelength axis.

    Parameters
    ----------
    recon_path : str
        Path to reconstruction file as numpy array in CYX format
    filter_path : str
        Path to filter data containing wavelengths
    peak_height : float
        Minimum height for peak detection
    threshold_ratio : float
        Ratio for horizontal threshold line
    recon_name : str
        Informative name of the reconstruction. If provided will
        be used as plot title
    out_path : str
        Path to output file prefix. If provided, figure will be saved
        with _spectral_profile suffix
    """
    if out_path:
        out_path_dir = os.path.dirname(out_path)
        os.makedirs(out_path_dir, exist_ok=True)
        out_name_suffix = pathlib.Path(out_path).stem

    # Load filter data
    filter_data = scipy.io.loadmat(filter_path)
    wavelengths = filter_data["waves"].squeeze()

    # Load and normalize reconstruction
    recon = np.load(recon_path).transpose(1, 2, 0).squeeze()[127, 127, :]
    recon = recon / recon.max()

    peaks, _ = find_peaks(recon, height=peak_height)

    plt.rcParams["font.sans-serif"] = "Arial"
    fig = plt.figure(dpi=150, figsize=(5, 5))
    plt.plot(wavelengths, recon, "r")
    plt.yticks([])

    if len(peaks) > 0:
        peak_idx = min(1, len(peaks) - 1)
        plt.axhline(
            y=recon[peaks[peak_idx]] * threshold_ratio, color="r", linestyle="--"
        )

    plt.xticks([450, 525, 600, 675, 750, 825], fontsize=24)
    plt.xlabel("Wavelength (nm)", fontsize=24)
    if recon_name:
        plt.title(recon_name)

    if out_path:
        with open(
            os.path.join(out_path_dir, out_name_suffix + "_spectral_profile.png"), "wb"
        ) as f:
            plt.savefig(f)
    return fig


def main():
    CROP_COORDS = {"vertical": (122, 135, 121, 134), "horizontal": (121, 134, 122, 135)}
    OUT_DIR = os.path.join(os.path.join(pathlib.Path(__file__).parent, "results"))

    configs = glob.glob(os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml"))
    print(f"Collected {len(configs)} configs:")
    for config_path in configs:
        print("\t", os.path.abspath(config_path))
    print("\n")

    for config_path in configs:
        config = helper.read_config(config_path)
        model_name = _get_unique_model_name(config)
        gt_filenames = sorted(os.listdir(config["base_data_path"]))
        recon_paths = [
            os.path.join(
                config["save_recon_path"],
                "_".join([model_name, pathlib.Path(gt_file).stem + ".npy"]),
            )
            for gt_file in gt_filenames
        ]
        assert all(
            os.path.exists(f) for f in recon_paths
        ), "Expected reconstruction not found."

        for recon_path, gt_filename in zip(recon_paths, gt_filenames):
            recon_name = os.path.splitext(gt_filename)[0]
            out_path = os.path.join(
                OUT_DIR, os.path.basename(os.path.dirname(recon_path)), recon_name
            )

            is_spatial = "spatial" in config["base_data_path"]

            peak_height = 0.5
            if "wavelength_714" in recon_path:
                peak_height = 0.25

            direction = "vertical"
            if direction not in config["base_data_path"]:
                direction = "horizontal"

            if is_spatial:
                recon_cropped = load_and_process_recon(
                    recon_path, CROP_COORDS[direction]
                )
                recon_crop, spatial_profile = plot_spatial_profile(
                    recon_cropped, direction, recon_name=recon_name, out_path=out_path
                )
                recon_crop.show()
                spatial_profile.show()
            else:
                spectral_profile = plot_spectral_profile(
                    recon_path,
                    config["mask_dir"],
                    peak_height=peak_height,
                    recon_name=recon_name,
                    out_path=out_path,
                )
                spectral_profile.show()


if __name__ == "__main__":
    main()
