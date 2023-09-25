# utils for experimental psfs
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import glob
from PIL import Image

# import torch
import os


def mean_denoised_image(psfs, pos, im_type, no_subtraction=False):
    # get relevant noise measurements
    psf_names = [psf_path.split("/")[-1] for psf_path in psfs]
    positions = list(set(["_".join(psf.split("_")[:2]) for psf in psf_names]))
    positions = [p for p in positions if p != pos]

    noise_images = [
        np.array(Image.open(psf))
        for psf in psfs
        if any([pos in psf for pos in positions])
        and all([characteristic in psf for characteristic in im_type])
    ]
    # get image
    im = [
        np.array(Image.open(psf))
        for psf in psfs
        if pos in psf and all([characteristic in psf for characteristic in im_type])
    ]

    # compute and subtract avg noise from image
    if len(noise_images) > 0:
        average_noise = np.mean(np.array(noise_images), 0)
    else:
        average_noise = np.zeros_like(im[0])

    if no_subtraction:
        return im[0]
    return im[0] - average_noise


def get_psfs(
    psf_dir="/home/cfoley_waller/defocam/defocuscamdata/calibration_data/psfs2",
    ext=".png",
    plot=False,
):
    psfs = glob.glob(os.path.join(psf_dir, "*" + ext))
    top_right_sharp = mean_denoised_image(psfs, pos="top_right", im_type=["sharp"])
    top_right_blurry_1 = mean_denoised_image(
        psfs, pos="top_right", im_type=["blurry_1"]
    )
    top_right_blurry_2 = mean_denoised_image(
        psfs, pos="top_right", im_type=["blurry_2", "300"]
    )

    top_left_sharp = mean_denoised_image(psfs, pos="top_left", im_type=["sharp"])
    top_left_blurry_1 = mean_denoised_image(psfs, pos="top_left", im_type=["blurry_1"])
    top_left_blurry_2 = mean_denoised_image(
        psfs, pos="top_left", im_type=["blurry_2", "300"]
    )

    bottom_right_sharp = mean_denoised_image(
        psfs, pos="bottom_right", im_type=["sharp"]
    )
    bottom_right_blurry_1 = mean_denoised_image(
        psfs, pos="bottom_right", im_type=["blurry_1"]
    )
    bottom_right_blurry_2 = mean_denoised_image(
        psfs, pos="bottom_right", im_type=["blurry_2", "300"]
    )

    bottom_left_sharp = mean_denoised_image(psfs, pos="bottom_left", im_type=["sharp"])
    bottom_left_blurry_1 = mean_denoised_image(
        psfs, pos="bottom_left", im_type=["blurry_1"]
    )
    bottom_left_blurry_2 = mean_denoised_image(
        psfs, pos="bottom_left", im_type=["blurry_2", "300"]
    )
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        ax[0].imshow(top_right_sharp[:50, -50:])
        ax[1].imshow(top_right_blurry_1[:50, -50:])
        ax[2].imshow(top_right_blurry_2[:50, -50:])
        plt.suptitle("top right")
        plt.show()

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        ax[0].imshow(top_left_sharp[:50, :50])
        ax[1].imshow(top_left_blurry_1[:50, :50])
        ax[2].imshow(top_left_blurry_2[:50, :50])
        plt.suptitle("top left")
        plt.show()

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        ax[0].imshow(bottom_right_sharp[-50:, -50:])
        ax[1].imshow(bottom_right_blurry_1[-50:, -50:])
        ax[2].imshow(bottom_right_blurry_2[-50:, -50:])
        plt.suptitle("bottom right")
        plt.show()

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        ax[0].imshow(bottom_left_sharp[-50:, :50])
        ax[1].imshow(bottom_left_blurry_1[-50:, :50])
        ax[2].imshow(bottom_left_blurry_2[-50:, :50])
        plt.suptitle("bottom left")
        plt.show()
    out = [
        top_right_sharp[:50, -50:],
        top_right_blurry_1[:50, -50:],
        top_right_blurry_2[:50, -50:],
        top_left_sharp[:50, :50],
        top_left_blurry_1[:50, :50],
        top_left_blurry_2[:50, :50],
        bottom_right_sharp[-50:, -50:],
        bottom_right_blurry_1[-50:, -50:],
        bottom_right_blurry_2[-50:, -50:],
        bottom_left_sharp[-50:, :50],
        bottom_left_blurry_1[-50:, :50],
        bottom_left_blurry_2[-50:, :50],
    ]
    return out


def get_psfs_dmm_37ux178(
    psf_dir="/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/psfs_9_22_2023_noiseavg32",
    ext=".bmp",
    scale=False,
    center_crop_width=120,
    crop_shape="square",
    usefirst=False,
    blurstride=1,
):
    psfs = []
    filenames = glob.glob(os.path.join(psf_dir, "*" + ext))
    filenames.sort()
    filenames = filenames[::blurstride]

    # open and convert to numpy
    psfs = [np.array(Image.open(file)) for file in filenames]

    # if specified, crop all psfs to center of the in-focus psf
    offset = None
    if usefirst:
        _, offset = center_crop_psf(psfs[0], center_crop_width)

    # crop with offset
    if center_crop_width > 0:
        psfs = [
            center_crop_psf(psf, center_crop_width, crop_shape, offset)[0]
            for psf in psfs
        ]
    if scale:
        psfs = [psf / 255 for psf in psfs]

    return psfs


def get_psfs_new(
    psf_dir="/home/cfoley_waller/defocam/defocuscamdata/calibration_data/new_camera_psfs",
    ext=".png",
    scale=False,
    center_crop_width=120,
):
    psfs = []
    filenames = glob.glob(os.path.join(psf_dir, "*" + ext))
    filenames.sort()

    for name in filenames:
        # sharp always goes in front, otherwise sort (blurry1 < blurry_2 < etc...)
        if "sharp" in name.split("/")[-1]:
            psfs = [name] + psfs
        else:
            psfs.append(name)

    # open and convert to numpy
    psfs = [np.array(Image.open(psf)) for psf in psfs]
    if center_crop_width > 0:
        psfs = [center_crop_psf(psf, center_crop_width)[0] for psf in psfs]
    if scale:
        psfs = [psf / 255 for psf in psfs]

    return psfs


def get_circular_kernel(diameter):
    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

    return kernel


def center_crop_psf(psf, width, shape="square", center_offset=None):
    # pad to allow larger crop
    padding = (
        (psf.shape[0] // 2, psf.shape[0] // 2),
        (psf.shape[1] // 2, psf.shape[1] // 2),
    )
    psf = np.pad(psf, padding, mode="median")

    if center_offset is None:
        # blur for more accurate center
        kernel = get_circular_kernel(7)
        psf_conv = signal.correlate(psf, kernel, mode="same")

        # get coords of max
        max_index = np.argmax(psf_conv)
        max_coords = np.unravel_index(max_index, psf.shape)
    else:
        max_coords = center_offset

    # crop around max (always returns even, favors left&above)
    square_crop = psf[
        max_coords[0] - width // 2 : max_coords[0] + width // 2,
        max_coords[1] - width // 2 : max_coords[1] + width // 2,
    ]
    circle_crop = np.multiply(get_circular_kernel(width), square_crop)

    if shape == "square":
        return square_crop, max_coords
    elif shape == "circle":
        return circle_crop, max_coords
    else:
        raise AssertionError("unhandled crop shape")


def one_normalize(im):
    im = im - np.min(im)
    return im / np.max(im)


def center_pad_to_shape(psfs, shape, val=0):
    # expects stack of psfs in form (z,y,x)
    pad_func = lambda a, b: np.pad(a, ((b[0], b[1]), (b[2], b[3]), (b[4], b[5])))

    # pad y
    if psfs.shape[1] < shape[0]:
        diff = shape[0] - psfs.shape[1]
        assert (diff) % 2 == 0, "psf dims must be even"
        psfs = pad_func(psfs, (0, 0, diff // 2, diff // 2, 0, 0))

    # pad x
    if psfs.shape[2] < shape[1]:
        diff = shape[1] - psfs.shape[2]
        assert (diff) % 2 == 0, "psf dims must be even"
        psfs = pad_func(psfs, (0, 0, 0, 0, diff // 2, diff // 2))

    return psfs


def get_psf_stack(psf_dir, num_ims, mask_shape):
    # Psfs for old viavi and basler cameras
    # psfs = get_psfs(psf_dir)
    # psfs = get_psfs_new(psf_dir)

    stack_xy_shape = mask_shape[1:]
    psfs = get_psfs_dmm_37ux178(
        psf_dir,
        center_crop_width=min(stack_xy_shape),
        crop_shape="square",
        usefirst=True,
        blurstride=1,
    )

    # normalize and pad
    psfs = psfs[0:num_ims]
    psfs = [one_normalize(sel) for sel in psfs]

    psfs = np.array(
        [center_pad_to_shape(np.expand_dims(psf, 0), stack_xy_shape) for psf in psfs]
    )

    # return reformatted shape (current: z,y,x,? -> desired: z,y,x)
    return np.transpose(psfs[..., 0, :, :], (0, 1, 2))
