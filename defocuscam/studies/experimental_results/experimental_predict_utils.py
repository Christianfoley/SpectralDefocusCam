import glob
import logging
import numpy as np, torch, scipy.io as io, os, matplotlib.pyplot as plt, h5py, PIL.Image as Image, pathlib
import sys
from torch.nn import Module

sys.path.insert(0, "../..")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import defocuscam.utils.helper_functions as helper
import defocuscam.utils.diffuser_utils as diffuser_utils
import defocuscam.dataset.precomp_dataset as ds


def get_experimental_measurement_stack(
    measurements_dir: str,
    image_center: tuple[int, int],
    image_dims: tuple[int, int],
    patch_size: tuple[int, int],
    blur_levels: int,
    blurstride: int = 1,
    blur_start_idx=0,
):
    """
    Experimental measurements should be stored in a single directory per measurement
    stack, in individual 2D HW format. The order should match the PSFs of the model
    characterizing the level of focus used to acquire each measurement
    E.g:
        path/to/dir:
            - 0.bmp # sharp measurement
            - 1.bmp
            - 2.bmp # blurry measurement
        path/to/psfs:
            - 0.bmp # focused psf
            - 1.bmp
            - 2.bmp # defocused psf

    Parameters
    ----------
    measurements_dir : str
        Path to the directory containing num_psfs measurements
    image_center : tuple of int
        (x, y) coordinates of the image center.
    image_dims : tuple of int
        (width, height) of the region to crop from the raw measurmeent
    patch_size : tuple of int
        (width, height) of the size to downsample the raw measurement to
        to fit into our model
    blur_levels : int
        Number of total psfs taken and expected by the model
    blurstride : int
        If the model psfs do not match one-to-one, they may use a stride (e.g. match every 2
        levels of blur). Indicate the stride here. Defaults to 1
    blur_start_idx : int
        Start index for the slicing of the measurements.

    Returns
    -------
    np.ndarray
        3d (NYX) measurement stack
    """
    experimental_measurements = sorted(glob.glob(os.path.join(measurements_dir, "*.bmp")))
    assert len(experimental_measurements) > 0, f"No measurements found for {measurements_dir}"

    if blurstride < 0:  # only defocused psfs
        parsed_exp_measurements = experimental_measurements[-1:]
    else:
        parsed_exp_measurements = experimental_measurements[
            blur_start_idx : blur_levels * blurstride : blurstride
        ]

    if len(parsed_exp_measurements) != blur_levels:
        logging.warning(
            "Experimental measurements unable to be parsed to blur stride, "
            f"continuing with the first {blur_levels} measurements"
        )
        experimental_measurements = experimental_measurements[:blur_levels]
    else:
        experimental_measurements = parsed_exp_measurements

    # Load and preprocess measurements
    preprocessed_arrays = [
        preprocess_experimental_measurement(
            measurement_path,
            image_center=image_center,
            image_dims=image_dims,
            patch_size=patch_size,
        )
        for measurement_path in experimental_measurements
    ]
    return np.stack(preprocessed_arrays, axis=0)


def preprocess_experimental_measurement(
    bmp_meas_path: str,
    image_center: tuple[int, int],
    image_dims: tuple[int, int],
    patch_size: tuple[int, int],
):
    """
    Preprocess measurements taken from the experimental prototype.

    Parameters
    ----------
    bmp_meas_path : str
        Path to the .bmp image measurement
    image_center : tuple of int
        (u, x) coordinates of the image center.
    image_dims : tuple of int
        (height, width) of the region to crop from the raw measurmeent
    patch_size : tuple of int
        (height, width) of the size to downsample the raw measurement to
        to fit into our model

    Returns
    -------
    np.ndarray
        Preprocessed measurement image.
    """
    centery, centerx = image_center
    dimy, dimx = image_dims
    meas = np.array(Image.open(bmp_meas_path), dtype=float)[
        centery - dimy // 2 : centery + dimy // 2,
        centerx - dimx // 2 : centerx + dimx // 2,
    ]

    # downsample
    meas = diffuser_utils.pyramid_down(meas, patch_size)

    return meas


def predict_experimental(experimental_meas_stack: np.ndarray, model: Module):
    """
    Typically, prediction is handled in the simulation model. However, since we've
    moved to experiment, we need to patch a few things in. This is a convenience
    function that goes from numpy array to numpy array, returning a 0-1 normalized
    reconstruction

    Parameters
    ----------
    input : np.ndarray
        3d measurement stach (CYX)
    model : Module
        prediction/reconstruction model
    """
    model.eval()
    exp_meas_stack = ds.Normalize(0, 1)(
        torch.tensor(experimental_meas_stack).to(next(model.parameters()).device)[
            None, :, None, ...
        ]
    )
    pred = model(exp_meas_stack)[0].detach().cpu().numpy().transpose(1, 2, 0)
    normed_pred = helper.value_norm(pred)
    return normed_pred


def save_reconstructed_measurement(
    recon: np.ndarray,
    out_base_path: str,
    checkpoint_path: str,
    measurement_path: str,
) -> str:
    """Convenience function to save a recon to a derived path"""
    trained_weights_basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    measurement_basename = os.path.basename(measurement_path)
    out_path = os.path.join(
        out_base_path,
        "_".join([trained_weights_basename, measurement_basename]) + ".npy",
    )
    os.makedirs(out_base_path, exist_ok=True)
    np.save(out_path, recon)
    return out_path


def save_reconstructed_fc_image(
    recon: np.ndarray,
    out_base_path: str,
    checkpoint_path: str,
    measurement_path: str,
    scaling: tuple[int, int, int] = (1, 1, 1),
    fc_range: tuple[int, int] = (398, 862),
    use_band_average: bool = False,
    gamma: float = 0.8,
) -> tuple[Image, str]:  # type: ignore
    """
    Convenience function to construct a false color image from a reconstructed
    measurement and save it to a derived path.
    """
    trained_weights_basename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    measurement_basename = os.path.basename(measurement_path)
    out_path = os.path.join(
        out_base_path,
        "_".join(
            [
                trained_weights_basename,
                measurement_basename,
                f"scaling={scaling}",
                f"fc-range={fc_range}",
            ]
        )
        + ".png",
    )
    if use_band_average:
        recon_fc = helper.select_and_average_bands(recon, fc_range=fc_range, scaling=scaling)
    else:
        recon_fc = helper.fast_rgb_img_from_spectrum(recon, fc_range=fc_range, gamma=gamma)

    recon_fc_uint8 = (helper.value_norm(recon_fc) * 255).astype(np.uint8)
    recon_image = Image.fromarray(recon_fc_uint8)

    os.makedirs(out_base_path, exist_ok=True)
    recon_image.save(out_path)
    return recon_image, out_path
