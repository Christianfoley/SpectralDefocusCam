import torch.nn.functional as F
import os, tqdm
import numpy as np
import torch
import torch.optim
import scipy.io
import cv2
from scipy.ndimage import generic_filter

import models.rdmpy.blur as blur

import matplotlib.pyplot as plt


def fft_psf(model, h):
    h_complex = pad_zeros_torch(model, torch.complex(h, torch.zeros_like(h)))
    H = torch.fft.fft2(torch.fft.ifftshift(h_complex, dim=(-1, -2))).unsqueeze(1)
    return H


def fft_im(im):
    xc = torch.complex(im, torch.zeros_like(im))
    Xi = torch.fft.fft2(xc)
    return Xi


def quantize(img, bins=256):
    maxfn = torch.max
    if isinstance(img, np.ndarray):
        maxfn = np.max
    img = floor_div((img / (maxfn(img) + 2.2e-15)) * (bins - 1), 1)
    return img


def floor_div(input, K):
    rem = torch.remainder(input, K)
    out = (input - rem) / K
    return out


def pyramid_down(image, out_shape):
    """
    Pyramid (gaussian binned) downsampling
    """
    if image.shape[0] == out_shape[0] and image.shape[1] == out_shape[1]:
        return image
    closest_pyr = cv2.pyrDown(image, (image.shape[1] // 2, image.shape[0] // 2))
    return cv2.resize(
        closest_pyr, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_AREA
    )


def binning_down(image, out_shape):
    """
    Binned downsampling
    """
    if image.shape[0] == out_shape[0] and image.shape[1] == out_shape[1]:
        return image
    else:
        assert image.shape[0] % out_shape[0] == 0, "crop & patch dims must be multiples"
        assert image.shape[1] % out_shape[1] == 0, "crop & patch dims must be multiples"

    assert (
        image.shape[0] // out_shape[0] == image.shape[1] // out_shape[1]
    ), "binned downsample rate must be equivalent in both dimensions"

    ksize = (image.shape[0] // out_shape[0], image.shape[1] // out_shape[1])
    avgpool = torch.nn.AvgPool2d(ksize)

    return avgpool(torch.tensor(image[None, None, ...])).numpy()[0, 0]


def replace_outlier_with_local_median(
    meas, neighborhood_size=3, n_stds=4, high_outliers_only=False
):
    """
    Utility function for cleaning up outlier pixels in an image.
    Replaces pixels > n_stds standard deviations from global mean with their local median.

    Parameters
    ----------
    meas : np.ndarray
        2d input image (y,x).
    neighborhood_size : int
       size of local neighborhood in pixels, by default 3
    n_stds : float, 4
        number of stds from mean to classify as outlier

    Returns
    -------
    np.ndarray
        The cleaned 2D array with outliers replaced by the mean of their neighboring pixels.
        This array will have the same shape and type as the input 'meas' array.
    """
    mean, std = np.mean(meas), np.std(meas)

    def local_func(neighborhood):
        center = neighborhood[len(neighborhood) // 2]  # nbrhood=unrolled 2d window
        delta = center - mean if high_outliers_only else np.abs(center - mean)
        if delta > (n_stds * std) + mean:
            return np.median([n for n in neighborhood if n != center])
        return center

    footprint = np.ones((neighborhood_size, neighborhood_size))
    return generic_filter(meas, local_func, footprint=footprint, mode="nearest")


def img_interp1d(samples, vals, newvals, verbose=False, force_range_overlap=True):
    """
    Given a set of images sampled from underlying (assumed linear) function, and the input
    values of the samples, will resample function along the specified interp axis to
    the "newvals".

    Note: forcing a range overlap can cause some issues with true image quality. It is
    strongly recommended that the newvals range is within the oldvals range.

    Parameters
    ----------
    samples : np.ndarray
        4d square image stack to interpolate along (c, vals, n, n)
    vals : np.ndarray
        2d array containing the original sample values (c, vals)
    newvals : np.ndarray
        2d array containing the new sample values (c, newvals)
    force_range_overlap : bool, optional
        If true, forces the min and max of vals to the min and max of newvals to prevent
        an out of interpolation range error

    Returns
    -------
    np.ndarray
        4d square image stack with out samples, (c, newvals, n, n)
    """
    c, v, n, n = samples.shape
    vals, newvals = vals.copy(), newvals.copy()
    assert vals.shape[:2] == (c, v), "wrong number of values to samples"
    assert newvals.shape[0] == c, "wrong number of newvals to samples"

    interp = scipy.interpolate.interp1d
    out_samples = np.zeros((c, newvals.shape[-1], n, n))
    for j in tqdm.tqdm(list(range(c)), desc="Interpolating") if verbose else range(c):
        # force old value range to encompas new value range
        if force_range_overlap:
            min_idx, max_idx = np.argmin(vals[j]), np.argmax(vals[j])
            vals[j, min_idx] = np.min(newvals[j])
            vals[j, max_idx] = np.max(newvals[j])
        interp = scipy.interpolate.interp1d(vals[j], samples[j], axis=0)
        out_samples[j] = interp(newvals[j])

    return out_samples


def rescale_to_psfs(batch, yx_shape, mode="trilinear"):
    """
    Rescales batch data via 3d bilinear interpolation to match the spatial
    dims of the psf shape

    Parameters
    ----------
    batch : torch.Tensor
        5d tensor of shape (batch, n_blur, channel, y, x)
    xy_shape : tuple
        shape desired yx_shape
    mode : str
        interpolation mode

    Returns
    -------
    tuple(torch.Tensor, tuple)
        rescaled batch and the "output size"
    """
    original_size = batch.shape[3:]

    if yx_shape == original_size:
        return batch, original_size

    batch = F.interpolate(batch, (batch.shape[2],) + yx_shape, mode=mode)
    return batch, original_size


def batch_ring_convolve(batch, psfs, device=torch.device("cpu")):
    """
    Performs ring convolution batch-wise with stack of RoFT PSFS and a batched stack
    of images

    Note that if the psf batch is too large in r and h, the batch's x and y dimensions
    will be upsampled to match

    Note that the method will attempt to broadcast along the n_blur dimension of the
    batch if it is of size 1

    Parameters
    ----------
    batch : torch.Tensor
        5d tensor of shape (batch, n_blur or 1, channel, y, x)
    psfs : torch.Tensor
        4d tensor of shape (n_blur, r, theta, h)
    device : torch.device, optional
        device, by default cpu

    Returns
    -------
    torch.Tensor
        5d blurred batch of shape (batch, n_blur, channel, y, x)
    """
    assert len(batch.shape) == 5, "batch must be of shape b, n, c, y, x"
    assert len(psfs.shape) == 4, "psf data must be of shape n, r, theta, h"

    # infer broadcasting along blur dimension
    (_, b_n, _, _, _), (n_b, _, _, _) = batch.shape, psfs.shape
    batch = batch.type(torch.float32)
    if b_n != psfs.shape[0]:
        assert b_n == 1, f"Blur dim of batch {b_n} does not match num psfs {n_b}"
        batch = batch.repeat(1, psfs.shape[0], 1, 1, 1)

    # fix size mismatch issue (we upsample to psfs->conv->downsample back)
    batch, output_size = rescale_to_psfs(batch, (psfs.shape[1], psfs.shape[3]))

    # batch-wise convolution for each blur level
    convolved_batch = torch.zeros_like(batch, device=device)
    for n in range(convolved_batch.shape[1]):
        convolved_batch[:, n, ...] = blur.batch_ring_convolve(
            batch[:, n, ...], psfs[n], device=device
        )

    # rescale to original input shape
    batch, _ = rescale_to_psfs(convolved_batch, output_size)

    del convolved_batch
    return batch


def preprocess_meas(
    meas, center, crop_size, dim, outlier_std_threshold=2.5, defective_pixels=[]
):
    """
    Preprocesses experimental measurements by cropping, normalizing, removing outlier
    and defective pixels, and zeroing

    Parameters
    ----------
    meas : np.ndarray
        2d experimental measurement
    center : tuple
        coordinates of the image center
    crop_size : tuple
        size of image in y and x
    dim : tuple
        output size of image in y and x
    outlier_std_threshold : float, optional
        number of stds away from median defining an outlier pixel, by default 2.5
    defective_pixels : list, optional
        list of coorinates of defective pixels, by default []

    Returns
    -------
    np.ndarray
        processed measurement patch
    """
    # crop
    crop = lambda im: im[
        center[0] - crop_size[0] // 2 : center[0] + crop_size[0] // 2,
        center[1] - crop_size[1] // 2 : center[1] + crop_size[1] // 2,
    ]
    meas = crop(meas)

    # pixels outside (outlier_std_threshold) * img std are outliers
    meas = replace_outlier_with_local_median(
        meas, n_stds=outlier_std_threshold, high_outliers_only=True
    )

    if defective_pixels:
        meas[defective_pixels] = 0

    # downsample
    meas = pyramid_down(meas, dim)

    # normalize
    one_norm = lambda im: (im - np.min(im)) / np.max(im - np.min(im))
    meas = one_norm(meas)

    return meas


def power_norm_mask(mask, start, end, norm_file="", norm_section=None):
    """
    Normalizes mask according to csv file power values.
    (rows of wavelength, power) where power is within the range (0,1)

    Parameters
    ----------
    mask : np.ndarray
        HS data cube (L,Y,X)
    start : int
        beginning wavelength of mask (in nm)
    end : int
        ending wavelength of mask (in nm)
    norm_file : str, optional
        path to normalization csv file, if None will normalize by average
        power over non-filtered section

    Returns
    -------
    np.ndarray
        Mask scaled by its power value
    """
    if norm_file:
        norm_data = np.genfromtxt(norm_file, delimiter=",")
        power_wavs, power = norm_data[1:, 0], norm_data[1:, 1]

    mask = np.copy(mask).astype(np.float32)
    wavs = np.linspace(start, end, mask.shape[0])

    ns = norm_section
    scales = np.mean(mask[:, ns[0] : ns[1], ns[2] : ns[3]], axis=(-1, -2))
    scales / np.max(scales)
    print(scales)
    for i in range(mask.shape[0]):
        # scale by closest value in the power spectrum
        if norm_file:
            scale = power[np.argmin(np.abs(power_wavs - wavs[i]))]
        else:
            scale = scales[i]
        mask[i] = (1 / scale) * mask[i]

    return mask


def mov_avg_mask(cube, waves, start, end, step=8, width=8):
    """
    Averages hyperspectral data cube across spectral channels with a
    moving average filter of step "step" (in lambda) and width "idx_width"
    (in indices)

    Parameters
    ----------
    cube : np.ndarray
        HS data cube (L,Y,X)
    waves : np.ndarray
        list of wavelengths corresponding to HS datacube
    start : int
        start wavelength (in lambda/nm)
    end : int
        end wavelength (in lambda/nm)
    step : int, optional
        step size (in lambda/nm), by default 8
    width : int, optional
        average filter width (in lambda/nm), by default 5

    Returns
    -------
    np.ndarray
        new data cube filtered over lambda (L_hat, Y, X)
    """
    idx_width = int(width // (waves[1] - waves[0]))
    new_waves = np.arange(start, end, step)

    lp_cube = np.zeros((len(new_waves),) + cube.shape[1:])
    actual_new_waves = np.zeros_like(new_waves)
    for i, wave in enumerate(new_waves):
        idx = np.where(waves == wave)[0][0]
        lp_cube[i] = np.mean(
            cube[
                max(0, idx - idx_width // 2) : min(
                    idx + idx_width // 2 + 1, len(waves) - 1
                )
            ],
            0,
        )
        actual_new_waves[i] = np.mean(
            waves[
                max(0, idx - idx_width // 2) : min(
                    idx + idx_width // 2 + 1, len(waves) - 1
                )
            ]
        )

    # safety check that the given wavelengths apply to the data cube
    assert np.all(
        np.equal(new_waves, actual_new_waves)
    ), "Detected mismatch between generated and requested wavelengths"

    return lp_cube, new_waves


def load_mask(
    path,
    patch_crop_center=[1050, 2023],  # [200, 2100],
    patch_crop_size=[768, 768],
    patch_size=[256, 256],
    old_calibration=False,
    sum_chans_2=False,
    downsample="binned",
):
    """
    Loads mask from saved .mat path

    Parameters
    ----------
    path : str
        path to spectral filter calibration mask
    patch_crop_center : list, optional
        center (in xy) of usable section, by default [1050, 2023]
    patch_crop_size : list, optional
        size (in xy) of usable section, by default [768, 768]
    patch_size : list, optional
        final/downsampled size (in xy) of usable seciton, by default [256, 256]

    Returns
    -------
    np.ndarray
        3d spectral filter calibration matrix
    """
    spectral_mask = scipy.io.loadmat(path)
    nm = lambda x: (x - np.min(x)) / (np.max(x - np.min(x)))
    dims = (
        patch_crop_center[0] - patch_crop_size[0] // 2,
        patch_crop_center[0] + patch_crop_size[0] // 2,
        patch_crop_center[1] - patch_crop_size[1] // 2,
        patch_crop_center[1] + patch_crop_size[1] // 2,
    )
    mask = spectral_mask["mask"][dims[0] : dims[1], dims[2] : dims[3]]

    if old_calibration:  # weird things about models trained with old calibration matrix
        mask = mask[100 : 100 + patch_size[0], 100 : 100 + patch_size[1], :-1]
        mask = (mask[..., 0::2] + mask[..., 1::2]) / 2
        mask = mask[..., 0:30]
    else:
        # widening "filter" resolution to adjascent wavelengths
        if sum_chans_2:
            mask1 = mask[..., 0:-1:2]
            mask2 = mask[..., 1::2]
            mask = (mask1 + mask2)[..., 0:30]

        # mask = mask / np.sum(
        #     mask, axis=-1, keepdims=True
        # )  # L1-norm each filter transmission

        # downsample & global normalize
        ds = binning_down if downsample == "binned" else pyramid_down
        mask = np.stack(
            [ds(mask[..., i], patch_size) for i in range(mask.shape[-1])],
            axis=-1,
        )

        # # clean bad pixels
        for i in tqdm.tqdm(list(range(mask.shape[-1])), desc="cleaning outliers"):
            mask[..., i] = replace_outlier_with_local_median(mask[..., i], 11, 7, True)

        nm = lambda x: (x - np.min(x)) / (np.max(x - np.min(x)))
    return nm(mask)


def process_mat_files(input_directory, output_directory, overwrite=True):
    """
    Process .mat files from the input directory, extracting the 'image' array
    and saving it as a new .mat file in the output directory.

    Parameters
    ----------
    input_directory : str
        Path to the directory containing .mat files.

    output_directory : str
        Path to the directory where the processed files will be saved.

    overwrite : bool, optional
        Controls whether to overwrite existing files in the output directory. Defaults to True.

    Notes
    -----
    - setting 'overwrite' to true allows file or entire directory overwriting, be careful!
    """
    if input_directory == output_directory and not overwrite:
        print("Cannot overwrite same directory unless overwrite flag is true")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = os.listdir(input_directory)

    for file in tqdm.tqdm(files):
        if file.endswith(".mat"):
            file_path = os.path.join(input_directory, file)
            output_file_path = os.path.join(output_directory, file)

            # Check if the file already exists in the output directory
            if os.path.exists(output_file_path) and not overwrite:
                print(
                    f"File {file} already exists in the output directory. Skipping..."
                )
                continue
            try:
                data = scipy.io.loadmat(file_path)
            except Exception as e:
                print(f"Failed to load {file_path}\n{e}")
                continue

            if "image" in data:
                image_array = data["image"]
                new_dict = {"image": image_array}

                scipy.io.savemat(output_file_path, new_dict)


def flip_channels(image):
    image_color = np.zeros_like(image)
    image_color[:, :, 0] = image[:, :, 2]
    image_color[:, :, 1] = image[:, :, 1]
    image_color[:, :, 2] = image[:, :, 0]
    return image_color


def pad_zeros_torch(model, x):
    PADDING = (
        model.PAD_SIZE1 // 2,
        model.PAD_SIZE1 // 2,
        model.PAD_SIZE0 // 2,
        model.PAD_SIZE0 // 2,
    )
    return F.pad(x, PADDING, "constant", 0)


def crop(model, x):
    C01 = model.PAD_SIZE0
    C02 = model.PAD_SIZE0 + model.DIMS0  # Crop indices
    C11 = model.PAD_SIZE1
    C12 = model.PAD_SIZE1 + model.DIMS1  # Crop indices
    return x[:, :, C01:C02, C11:C12]


def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def ifftshift2d(x):
    for dim in range(len(x.size()) - 1, 0, -1):
        x = roll_n(x, axis=dim, n=x.size(dim) // 2)

    return x  # last dim=2 (real&imag)


def roll_n(X, axis, n):
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def get_radius(y, x):
    return (y**2 + x**2) ** (1 / 2)


###### Complex operations ##########
def complex_multiplication(t1, t2):
    real1, imag1 = torch.unbind(t1, dim=-1)
    real2, imag2 = torch.unbind(t2, dim=-1)
    return torch.stack(
        [real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1
    )


def complex_abs(t1):
    real1, imag1 = torch.unbind(t1, dim=2)
    return torch.sqrt(real1**2 + imag1**2)


def make_real(c):
    out_r, _ = torch.unbind(c, -1)
    return out_r


def make_complex(r, i=0):
    if i == 0:
        i = torch.zeros_like(r, dtype=torch.float32)
    return torch.stack((r, i), -1)


def tv_loss(x, beta=0.5):
    """Calculates TV loss for an image `x`.
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    """
    x = x.cuda()
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)
    return torch.sum(dh[:, :, :-1] + dw[:, :, :, :-1])


def my_pad(model, x):
    PADDING = (
        model.PAD_SIZE1 // 2,
        model.PAD_SIZE1 // 2,
        model.PAD_SIZE0 // 2,
        model.PAD_SIZE0 // 2,
    )
    return F.pad(x, PADDING, "constant", 0)


def crop_forward(model, x):
    C01 = model.PAD_SIZE0 // 2
    C02 = model.PAD_SIZE0 // 2 + model.DIMS0  # Crop indices
    C11 = model.PAD_SIZE1 // 2
    C12 = model.PAD_SIZE1 // 2 + model.DIMS1  # Crop indices
    return x[..., C01:C02, C11:C12]
