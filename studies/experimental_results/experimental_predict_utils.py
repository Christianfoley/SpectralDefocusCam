import glob
import logging
import numpy as np, torch, scipy.io as io, os, matplotlib.pyplot as plt, h5py, PIL.Image as Image, pathlib
import sys
from torch.nn import Module

sys.path.insert(0, "../..")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import utils.helper_functions as helper
import utils.diffuser_utils as diffuser_utils
import dataset.preprocess_data as prep_data
import dataset.precomp_dataset as ds

import train


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
    experimental_measurements = sorted(
        glob.glob(os.path.join(measurements_dir, "*.bmp"))
    )
    assert (
        len(experimental_measurements) > 0
    ), f"No measurements found for {measurements_dir}"

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
        recon_fc = helper.select_and_average_bands(
            recon, fc_range=fc_range, scaling=scaling
        )
    else:
        recon_fc = fast_rgb_img_from_spectrum(recon, fc_range=fc_range, gamma=gamma)

    recon_fc_uint8 = (helper.value_norm(recon_fc) * 255).astype(np.uint8)
    recon_image = Image.fromarray(recon_fc_uint8)

    os.makedirs(out_base_path, exist_ok=True)
    recon_image.save(out_path)
    return recon_image, out_path


def resample_spectral_cube(cube, wavelengths, step=10):
    """
    Resample a hyperspectral cube along the spectral axis to a uniform wavelength grid.

    Parameters:
    - cube: ndarray of shape (H, W, λ), the original hyperspectral data.
    - wavelengths: 1D array of shape (λ,), corresponding to the spectral dimension.
    - step: desired wavelength step size in nm (e.g., 10 for 10nm intervals).

    Returns:
    - new_cube: ndarray of shape (H, W, λ'), resampled spectral cube.
    - new_wavelengths: 1D array of shape (λ'), uniformly spaced wavelengths.
    """
    wavelengths = np.asarray(wavelengths)
    H, W, L = cube.shape

    if L != len(wavelengths):
        raise ValueError("Last dimension of cube must match length of wavelengths.")

    # Create new wavelength axis
    min_wl = np.ceil(wavelengths.min())
    max_wl = np.floor(wavelengths.max())
    new_wavelengths = np.arange(min_wl, max_wl + 1e-6, step)

    # Flatten spatial dimensions
    flat_cube = cube.reshape(-1, L)

    # Interpolate each spectrum
    new_flat_cube = np.array(
        [np.interp(new_wavelengths, wavelengths, spectrum) for spectrum in flat_cube]
    )

    # Reshape back to (H, W, λ')
    new_cube = new_flat_cube.reshape(H, W, -1)

    return new_cube, new_wavelengths


def fast_rgb_img_from_spectrum(data_cube, fc_range, step=10, gamma=0.7):
    """
    Convert a hyperspectral data cube to an RGB image using vectorized colour-science.

    Parameters:
    - data_cube: np.ndarray of shape (H, W, λ)
    - fc_range: tuple of (start_wavelength, end_wavelength) in nm
    - step: target wavelength resampling interval in nm
    - gamma: gamma correction value

    Returns:
    - RGB image (H, W, 3)
    """
    import numpy as np
    import colour
    from colour import SpectralShape, MSDS_CMFS, SDS_ILLUMINANTS

    H, W, L = data_cube.shape
    wavs = np.linspace(fc_range[0], fc_range[1], L)
    cube, new_wavs = resample_spectral_cube(data_cube, wavs, step)

    # Ensure data is in reasonable range (normalize if needed)
    # Assuming your spectral data represents reflectance (0-1) or radiance
    if np.max(cube) > 10:  # If data seems to be in large units
        cube = cube / np.max(cube)  # Normalize to 0-1 range

    # Flatten spatial dimensions
    pixels = cube.reshape(-1, cube.shape[-1])  # shape: (H*W, L)

    # Create spectral shape
    wavelength_interval = new_wavs[1] - new_wavs[0] if len(new_wavs) > 1 else step
    shape = SpectralShape(
        start=new_wavs[0], end=new_wavs[-1], interval=wavelength_interval
    )

    # Get CMFs and align to our wavelengths
    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    cmfs_interp = cmfs.copy().align(shape)
    cmf_array = cmfs_interp.values  # shape: (L, 3) for X, Y, Z

    # Get illuminant and align to same wavelengths
    illuminant = SDS_ILLUMINANTS["D65"].copy().align(shape)
    illum_array = illuminant.values.flatten()  # Ensure 1D

    # Proper normalization constant (standard CIE calculation)
    # k normalizes so that perfect white reflector gives Y=100
    k = 100.0 / np.trapz(cmf_array[:, 1] * illum_array, dx=wavelength_interval)

    # For reflectance data, multiply by illuminant
    # If your data is already radiance, you might skip illuminant multiplication
    illuminated_pixels = pixels * illum_array[np.newaxis, :]  # Broadcast illuminant

    # Integrate to get XYZ values
    # Using trapezoidal integration approximation with matrix multiplication
    XYZ = k * wavelength_interval * np.dot(illuminated_pixels, cmf_array)

    # Ensure XYZ values are reasonable
    # print(
    #     f"XYZ range: X[{np.min(XYZ[:, 0]):.3f}, {np.max(XYZ[:, 0]):.3f}], "
    #     f"Y[{np.min(XYZ[:, 1]):.3f}, {np.max(XYZ[:, 1]):.3f}], "
    #     f"Z[{np.min(XYZ[:, 2]):.3f}, {np.max(XYZ[:, 2]):.3f}]"
    # )

    # Convert XYZ to linear sRGB (no gamma correction yet)
    rgb_linear = colour.XYZ_to_sRGB(XYZ / 100.0)  # This gives linear RGB values

    # Handle out-of-gamut colors more gracefully
    rgb_linear = np.clip(rgb_linear, 0, 1)

    # Apply custom gamma correction for better appearance
    gamma_corrected = np.power(rgb_linear, 1.0 / gamma)

    # Final clipping
    rgb_final = np.clip(gamma_corrected, 0, 1)

    # print(f"Final RGB range: [{np.min(rgb_final):.3f}, {np.max(rgb_final):.3f}]")

    # Reshape to image
    return rgb_final.reshape(H, W, 3)
