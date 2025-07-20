import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional
import json

import tqdm
from utils.helper_functions import (
    fast_rgb_img_from_spectrum,
    _resample_spectral_cube,
    value_norm,
)

GIF_FPS = 12
GIF_INTERPOLATION_FACTOR = 2  # multiplier on number of channels for smooth scanningn
OUT_DIR = "/home/cfoley/SpectralDefocusCam/visualization/outputs/"


def _get_sim_samples_from_directory(
    sim_sample_out_dir: str,
    spectral_range: tuple,
    gamma: float,
    out_dir=OUT_DIR,
):
    """
    Builds a pipeline input config dictionary, given a path to a json dictionary of the structure:
    {
        sample_name: {
            pred_type: pred_path,
            pred2_type: pred2_path,
            ...
        }
    }
    """
    directory_path = os.path.join(sim_sample_out_dir, "sim_preds_directory.json")
    assert os.path.exists(directory_path)

    with open(directory_path, "r") as f:
        directory = json.load(f)

    samples = {}
    for sample_name, sample_predictions in directory.items():
        for prediction_type, prediction_path in sample_predictions.items():
            name = f"{os.path.splitext(sample_name)[0]}_{prediction_type}"
            samples[name] = {
                "sample_path": prediction_path,
                "spectral_range": spectral_range,
                "gamma": gamma,
                "out_dir": out_dir,
            }
    return samples


def get_rgb_of_datacube_slice(datacube, slice_index, fc_range, gamma):
    """Get RGB image from a single slice of the hyperspectral datacube."""
    slice_cube = np.zeros_like(datacube)
    slice_cube[:, :, slice_index] = datacube[:, :, slice_index]
    rgb_frame = fast_rgb_img_from_spectrum(
        slice_cube,
        fc_range,
        gamma=gamma,
    )
    return rgb_frame


def _generate_spectral_scan_frames(
    data_cube: np.ndarray,
    show_fc: bool = False,
    fc_range: Optional[Tuple[int, int]] = None,
    gamma: float = 0.7,
) -> list:
    """
    Generate frames for a spectral scan of a hyperspectral datacube.
    """
    H, W, L = data_cube.shape
    maxvalue = np.max(data_cube)
    frames = []
    for i in tqdm.tqdm(range(L)):
        if show_fc:
            rgb_frame = get_rgb_of_datacube_slice(data_cube, i, fc_range, gamma)
        else:
            rgb_frame = data_cube[:, :, i]

        frame = Image.fromarray(
            (np.clip(rgb_frame / maxvalue, 0, 1) * 255).astype(np.uint8)
        )
        frames.append(frame)

    # Backward scan (excluding endpoints to avoid duplicate frames)
    for i in tqdm.tqdm(range(L - 2, 0, -1)):
        if show_fc:
            rgb_frame = get_rgb_of_datacube_slice(data_cube, i, fc_range, gamma)
        else:
            rgb_frame = data_cube[:, :, i]

        frame = Image.fromarray(
            (np.clip(rgb_frame / maxvalue, 0, 1) * 255).astype(np.uint8)
        )
        frames.append(frame)
    return frames


def _draw_red_scanline(image, position, width, maxvalue):
    """
    Draw a red scanline on the image at the specified position and width.
    """
    image = image.copy()

    # Assumes axis is 1 for L
    channel_slice = slice(
        max(position - width // 2, 0),
        min(position + width // 2 + 1, image.shape[1]),
    )

    image[:, channel_slice, 0] = maxvalue
    image[:, channel_slice, 1:] = 0
    return image


def _generate_side_profile_frames(
    data_cube: np.ndarray,
    profile_side: str = "top",
    show_fc: bool = False,
    fc_range: Optional[Tuple[int, int]] = None,
    gamma: float = 0.7,
    scanline_width: int = 3,
) -> list:
    """
    Generate frames for a side profile of the hyperspectral datacube.
    """
    assert profile_side in ["top", "right"], "profile_side must be 'top' or 'right'"
    maxvalue = np.max(data_cube)
    H, W, L = data_cube.shape

    # Generate the base frame for the profile. Since we're looking from the "top" or "right",
    # the "top" rgb cube is W X L X 3 or W X L X 1 for greyscale, and the "right" rgb cube is
    # H X L X 3 or H X L X 1 for greyscale.
    if profile_side == "top":
        profile_data_slice = data_cube[0, :, :]
    elif profile_side == "right":
        profile_data_slice = data_cube[:, -1, :]

    if show_fc:
        assert fc_range is not None, "fc_range must be provided for wavelength color"
        rgb_slice = np.zeros([*profile_data_slice.shape[:2], 3])
        for i in range(L):
            rgb_column = get_rgb_of_datacube_slice(
                np.expand_dims(profile_data_slice, axis=0),
                i,
                fc_range,
                gamma,  # us an imaginary spatial dim
            )
            rgb_slice[:, i, :] = rgb_column[0, :, :]
    else:
        rgb_slice = np.stack([profile_data_slice] * 3, axis=-1)

    # Now, generate the red line scanning across the L of the RGB slice, indicating the
    # wavelength position at a given framme.
    frames = []
    for i in tqdm.tqdm(range(rgb_slice.shape[1])):
        scanline_frame = _draw_red_scanline(rgb_slice, i, scanline_width, maxvalue)

        # since we view the top horizontally, rotate it 90 degrees
        if profile_side == "top":
            scanline_frame = np.rot90(scanline_frame, axes=(0, 1))
        frames.append(
            Image.fromarray(
                (np.clip(scanline_frame / maxvalue, 0, 1) * 255).astype(np.uint8)
            )
        )
    # Backward scan (excluding endpoints to avoid duplicate frames)
    for i in tqdm.tqdm(range(rgb_slice.shape[1] - 2, 0, -1)):
        scanline_frame = _draw_red_scanline(rgb_slice, i, scanline_width, maxvalue)

        if profile_side == "top":
            scanline_frame = np.rot90(scanline_frame, axes=(0, 1))
        frames.append(
            Image.fromarray(
                (np.clip(scanline_frame / maxvalue, 0, 1) * 255).astype(np.uint8)
            )
        )

    return frames


def create_spectral_scan_gif(
    data_cube: np.ndarray,
    spectral_range: Optional[Tuple[int, int]],
    output_path: str,
    show_fc: bool = False,
    fc_gamma: float = 0.7,
    fps: int = 30,
    interpolate_factor: Optional[int] = 1,
    top_profile=True,
    right_profile=True,
) -> None:
    """Create an aninpyed GIF that scans through a hyperspectral datacube,
    showing only one channel at a time, but always using the full datacube shape.

    Parameters
    ----------
    data_cube : np.ndarray
        Hyperspectral data cube of shape (H, W, λ)
    spectral_range : Tuple[int, int]
        (min_wavelength, max_wavelength) in nm
    output_path : str
        Path to save the output GIF
    show_fc : bool, optional
        If True, color the frames based on wavelength; if False, use grayscale,
        by default True
    fc_gamma : float, optional
        Gamma correction value for RGB conversion, by default 0.7
    fps : int, optional
        frames-per-second of the video, by default 30
    interpolate_factor : Optional[int], optional
        Factor by which to interpolate the frames for smoother transitions, by default 1
    top_profile : bool, optional
        If True, also generate a top profile of the data cube as a gif with a line scan
        passing over the channels. By default True
    right_profile : bool, optional
        If True, also generate a right profile of the data cube as a gif with a line scan
        passing over the channels. By default True
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    assert os.path.splitext(output_path)[1] == ".gif", "Output path must end with .gif"
    frame_duration = 1000 / fps

    # Linearly interpolate the cube to make it smoother
    if interpolate_factor is not None and interpolate_factor > 1:
        assert isinstance(
            interpolate_factor, int
        ), "interpolate_factor must be an integer"
        wavs = np.linspace(spectral_range[0], spectral_range[1], data_cube.shape[-1])
        wavstep = (wavs[1] - wavs[0]) / interpolate_factor
        data_cube, new_wavs = _resample_spectral_cube(data_cube, wavs, step=wavstep)
        print(f"Resampled wavelengths ({len(new_wavs)}): {new_wavs}")

    # Generate and save gifs
    scan_frames = _generate_spectral_scan_frames(
        data_cube,
        show_fc=show_fc,
        fc_range=spectral_range,
        gamma=fc_gamma,
    )
    scan_frames[0].save(
        output_path,
        save_all=True,
        append_images=scan_frames[1:],
        duration=frame_duration,
        loop=0,
    )
    print(f"Saved spectral scan GIF to {output_path}")

    if top_profile:
        top_profile_frames = _generate_side_profile_frames(
            data_cube,
            fc_range=spectral_range,
            show_fc=show_fc,
            profile_side="top",
            gamma=fc_gamma,
        )
        top_profile_output_path = output_path.replace(".gif", "_top_profile.gif")
        top_profile_frames[0].save(
            top_profile_output_path,
            save_all=True,
            append_images=top_profile_frames[1:],
            duration=frame_duration,
            loop=0,
        )
        print(f"Saved top profile GIF to {top_profile_output_path}")

    if right_profile:
        right_profile_frames = _generate_side_profile_frames(
            data_cube,
            fc_range=spectral_range,
            show_fc=show_fc,
            profile_side="right",
            gamma=fc_gamma,
        )
        right_profile_output_path = output_path.replace(".gif", "_right_profile.gif")
        right_profile_frames[0].save(
            right_profile_output_path,
            save_all=True,
            append_images=right_profile_frames[1:],
            duration=frame_duration,
            loop=0,
        )
        print(f"Saved right profile GIF to {right_profile_output_path}")


def run_visualization(
    sample_name: str,
    sample_path: str,
    spectral_range: tuple[int, int],
    out_dir: str = OUT_DIR,
    gamma=0.5,
):
    """Helper subrouting for generating a spectral scan gif and a false color image"""
    sample_data_cube = np.squeeze(np.load(sample_path))
    assert (
        len(sample_data_cube.shape) == 3
    ), f"Data cube must be H, W, L for this pipeline, got {sample_data_cube.shape}"

    create_spectral_scan_gif(
        sample_data_cube,
        spectral_range=spectral_range,
        output_path=os.path.join(out_dir, f"{sample_name}_spectral_scan.gif"),
        fps=GIF_FPS,
        interpolate_factor=GIF_INTERPOLATION_FACTOR,
        top_profile=True,
        right_profile=True,
    )

    Image.fromarray(
        (
            value_norm(
                fast_rgb_img_from_spectrum(
                    sample_data_cube,
                    fc_range=spectral_range,
                    gamma=gamma,
                )
            )
            * 255
        ).astype(np.uint8)
    ).save(os.path.join(out_dir, f"{sample_name}_fc.png"))


def main():
    # Example usage
    import sys

    sys.path.append("../..")  # Add project root to path

    # # ----------------- EXPERIMENTAL DATA ---------------- #
    samples = {
        "outside_nine2_campanile": {
            "sample_path": "/home/cfoley/SpectralDefocusCam/studies/experimental_results/outputs/saved_model_ep60_testloss_0.053416458687380604_outside_nine2.npy",
            "spectral_range": (398, 762),
            "gamma": 0.5,
        },
        "outside_eight2_umbrella": {
            "sample_path": "/home/cfoley/SpectralDefocusCam/studies/experimental_results/outputs/saved_model_ep60_testloss_0.053416458687380604_outside_eight2.npy",
            "spectral_range": (398, 862),
            "gamma": 0.5,
        },
        "outside_six_logo": {
            "sample_path": "/home/cfoley/SpectralDefocusCam/studies/experimental_results/outputs/saved_model_ep60_testloss_0.053416458687380604_outside_six.npy",
            "spectral_range": (398, 862),
            "gamma": 0.5,
        },
        "outside_3_author": {
            "sample_path": "/home/cfoley/SpectralDefocusCam/studies/experimental_results/outputs/saved_model_ep49_testloss_0.05782177185882693_outside_three.npy",
            "spectral_range": (398, 862),
            "gamma": 0.5,
        },
    }

    for sample_name, kwargs in tqdm.tqdm(
        samples.items(), desc="Running experimental vis..."
    ):
        run_visualization(sample_name=sample_name, **kwargs)

    # # ----------------- SIMULATION DATA (no noise)----------------- #
    # SIM_SAMPLE_DIR = "/home/cfoley/SpectralDefocusCam/visualization/results"
    # samples = _get_sim_samples_from_directory(
    #     SIM_SAMPLE_DIR, spectral_range=(360, 660), gamma=0.53
    # )
    # for sample_name, kwargs in tqdm.tqdm(
    #     samples.items(), desc="Running simulation vis..."
    # ):
    #     run_visualization(sample_name=sample_name, **kwargs)

    # # ----------------- SIMULATION DATA (noised)----------------- #
    # LOW_NOISE_SIM_SAMPLE_DIR = (
    #     "/home/cfoley/SpectralDefocusCam/visualization/results/low_noise"
    # )
    # samples = _get_sim_samples_from_directory(
    #     LOW_NOISE_SIM_SAMPLE_DIR,
    #     spectral_range=(360, 660),
    #     gamma=0.53,
    #     out_dir=os.path.join(OUT_DIR, "low_noise"),
    # )
    # for sample_name, kwargs in tqdm.tqdm(
    #     samples.items(), desc="Running simulation vis..."
    # ):
    #     run_visualization(sample_name=sample_name, **kwargs)

    # MED_NOISE_SIM_SAMPLE_DIR = (
    #     "/home/cfoley/SpectralDefocusCam/visualization/results/medium_noise"
    # )
    # samples = _get_sim_samples_from_directory(
    #     MED_NOISE_SIM_SAMPLE_DIR,
    #     spectral_range=(360, 660),
    #     gamma=0.53,
    #     out_dir=os.path.join(OUT_DIR, "medium_noise"),
    # )
    # for sample_name, kwargs in tqdm.tqdm(
    #     samples.items(), desc="Running simulation vis..."
    # ):
    #     run_visualization(sample_name=sample_name, **kwargs)

    # HIGH_NOISE_SIM_SAMPLE_DIR = (
    #     "/home/cfoley/SpectralDefocusCam/visualization/results/high_noise"
    # )
    # samples = _get_sim_samples_from_directory(
    #     HIGH_NOISE_SIM_SAMPLE_DIR,
    #     spectral_range=(360, 660),
    #     gamma=0.53,
    #     out_dir=os.path.join(OUT_DIR, "high_noise"),
    # )
    # for sample_name, kwargs in tqdm.tqdm(
    #     samples.items(), desc="Running simulation vis..."
    # ):
    #     run_visualization(sample_name=sample_name, **kwargs)


if __name__ == "__main__":
    main()
