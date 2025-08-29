# utils for experimental psfs
import sys
import glob
import os
import gc
from tqdm import tqdm

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import io, ndimage
from skimage import feature, morphology
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vision_F

import defocuscam.utils.diffuser_utils as diffuser_utils
from defocuscam.models.rdmpy._src import seidel, util, polar_transform


### ---------- Utility functions ---------- ##


def get_circular_kernel(diameter):
    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

    return kernel


def denoise_speckle(img, thresh_val=None):
    if thresh_val == None:
        thresh_val = np.median(img)

    # get binary de-speckling mask
    denoise_mask = cv2.medianBlur((img > thresh_val).astype("float32"), ksize=5)
    denoise_mask = denoise_mask > np.max(denoise_mask) / 2

    denoised_img = np.where(denoise_mask > 0, img, thresh_val)
    return denoised_img


def one_normalize(im):
    im = im - np.min(im)
    return im / np.max(im)


def thresh(psf, quantile=0.33, use_otsu=False):
    if use_otsu:
        blur = cv2.GaussianBlur(psf, (5, 5), 0)
        mask = cv2.threshold(
            blur,
            np.quantile(blur, quantile),
            np.max(blur),
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        return psf * mask
    else:
        mask = np.where(psf > np.quantile(psf, quantile), 1, 0)

    return psf * mask


def apply_mask_center(psf, width, shape="square"):
    if shape == "square":
        mask = np.ones((width, width))
    elif shape == "circle":
        mask = get_circular_kernel(width)

    mask = center_pad_to_shape(mask[None, ...], psf.shape)[0]

    return psf * mask


def simulate_gaussian_psf(dim, w_blur):
    """
    Simulate a gaussian psf

    Parameters
    ----------
    dim : int
        dimension of output
    w_blur : np.ndarray or list
        1d array of blur levels (arbitrary float)

    Returns
    -------
    np.ndarray
        stack of psfs (n_blur, dim, dim)
    """

    x = np.linspace(-1, 1, dim)
    y = np.linspace(-1, 1, dim)
    X, Y = np.meshgrid(x, y)

    psfs = []
    for i in range(0, len(w_blur)):
        var_yx = (w_blur[i], w_blur[i])
        psf = np.exp(-((X / var_yx[0]) ** 2 + (Y / var_yx[1]) ** 2))
        psfs.append(psf / np.linalg.norm(psf, ord=float("inf")))
    return np.stack(psfs, 0)


def process_psf_patch(patch, coords, threshold, psf_dim):
    """
    Utility for processing psf patches to remove background noise, center, and
    threshold values.

    Parameters
    ----------
    patch : np.ndarray
        input patch (2d numpy array) (y,x)
    coords : tuple
        coordinates of psf in input patch (to center around)
    threshold : float
        value between 0 and 1, quantile threshold value
    psf_dim : int or tuple
        dimension of output image patch, if int will treat as square

    Returns
    -------
    np.ndarray
        processed psf patch, centered around the psf coordinates
    """
    d0, d1 = psf_dim, psf_dim
    if not isinstance(psf_dim, int):
        d0, d1 = psf_dim

    patch = thresh(one_normalize(patch), quantile=threshold) * np.max(patch)

    patch = np.pad(patch, ((d0 // 2, d0 // 2), (d1 // 2, d1 // 2)))
    patch = patch[
        coords[0] : coords[0] + d0,
        coords[1] : coords[1] + d1,
    ]

    return patch


def center_pad_to_shape(psfs, shape):
    """
    Center pads stack of 2d images to shape

    Parameters
    ----------
    psfs : np.ndarray or torch.Tensor
        stack of psfs (z,y,x)
    shape : tuple
        y,x shape to pad to

    Returns
    -------
    np.ndarray or torch.Tensor
        padded image stack
    """
    # expects stack of psfs in form (z,y,x)
    if isinstance(psfs, np.ndarray):
        pad_func = lambda a, b: np.pad(a, ((b[0], b[1]), (b[2], b[3]), (b[4], b[5])))
    elif isinstance(psfs, torch.Tensor):
        pad_func = lambda a, b: F.pad(a, (b[5], b[4], b[3], b[2], b[1], b[0]))

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


def center_crop_to_shape(psfs, shape):
    """
    Center crops stack of 2d images to shape

    Parameters
    ----------
    psfs : np.ndarray or torch.Tensor
        stack of psfs (z,y,x)
    shape : tuple
        y,x shape to crop to

    Returns
    -------
    np.ndarray or torch.Tensor
        cropped image stack
    """
    # Calculate the differences in dimensions
    diff_y = psfs.shape[1] - shape[0]
    diff_x = psfs.shape[2] - shape[1]

    # Calculate crop amounts for top, bottom, left, and right
    crop_top = diff_y // 2
    crop_bottom = diff_y - crop_top
    crop_left = diff_x // 2
    crop_right = diff_x - crop_left

    # Perform cropping
    cropped_psfs = psfs[:, crop_top:-crop_bottom, crop_left:-crop_right]

    return cropped_psfs


def get_psf_center(
    psf,
    width=None,
    shape="square",
    center_offset=None,
    kernel_size=7,
    return_cropped=False,
):
    """
        Utility function to find the center coordinates of a psf in an image.
    If specified, will crop input image to center, and return cropped image also

    Parameters
    ----------
    psf : np.ndarray
        2d numpy array with one psf in view
    width : int, optional
        width of output if returning cropped, by default None
    shape : str, optional
        shape of output if returnning cropped, by default "square"
    center_offset : tuple, optional
        coordinates if psf center is already known and just cropping is desired,
        by default None
    kernel_size : int, optional
        size of kernel for center finding (approx size of psf), by default 7
    return_cropped : bool, optional
        whether to crop input image and return, by default False

    Returns
    -------
    np.ndarray
        cropped input (if specified)
    tuple
        psf center coordinates
    """
    # pad to allow larger crop
    padding = (
        (kernel_size, kernel_size),
        (kernel_size, kernel_size),
    )
    psf = np.pad(psf, padding)

    if center_offset is None:
        # blur for center of mass
        psf = psf * np.where(psf > np.quantile(psf, 0.75), psf, 0)
        psf_conv = cv2.GaussianBlur(psf, (kernel_size, kernel_size), 0)

        max_index = np.argmax(psf_conv)
        max_coords = np.unravel_index(max_index, psf.shape)
    else:
        max_coords = (
            center_offset[0] + padding[0][0],
            center_offset[1] + padding[1][0],
        )

    # crop around max (always returns even, favors left&above)
    if return_cropped:
        square_crop = psf[
            max_coords[0] - width // 2 : max_coords[0] + width // 2,
            max_coords[1] - width // 2 : max_coords[1] + width // 2,
        ]
        max_coords = (
            max_coords[0] - padding[0][0],
            max_coords[1] - padding[1][0],
        )

        if shape == "square":
            return square_crop, max_coords
        elif shape == "circle":
            circle_crop = apply_mask_center(square_crop, width=width, shape=shape)
            return circle_crop, max_coords
        else:
            raise NotImplementedError("unhandled crop shape")
    else:
        max_coords = (
            max_coords[0] - padding[0][0],
            max_coords[1] - padding[1][0],
        )
        return max_coords


def even_exposures(psfs, blur_levels, exposures, verbose=False):
    """
    Applies a value scaling to psfs taken with different exposure
    levels to increase snr

    Parameters
    ----------
    psfs : list(np.ndarray)
        list of psfs (2d arrays)
    blur_levels : int
        number of focus levels
    exposures : list(float)
        list of exposure levels
    """
    assert len(exposures) == blur_levels, "All focus levels must have exposure level"
    base_exposure = exposures[0]
    scaling_vals = [base_exposure / exp for exp in exposures]

    scaled_psfs = []
    for i, psf in enumerate(psfs):
        scaled_psfs.append(psf * scaling_vals[i % len(scaling_vals)])

    if verbose:
        print(f"Scaled psfs by values: {scaling_vals}")
    return scaled_psfs


def read_psfs(psf_dir, crop=None, patchsize=None, verbose=False):
    """
    Reads in ordered psf measurements stored as .bmp or .tiff files from a directory.

    Parameters
    ----------
    psf_dir : str
        directory containing measurements
    crop : tuple, optional
        cropping tuple: (y1, x1, y1, x2), by default None
    patchsize : tuple, optional
        output size (using pyramind down) to resize to, by default None

    Returns
    -------
    list
        list of psf measurements as numpy arrays
    """
    pathlist = sorted(glob.glob(os.path.join(psf_dir, "*.bmp")))
    pathlist += sorted(glob.glob(os.path.join(psf_dir, "*.tiff")))
    psfs = []
    assert len(pathlist) > 0, f"No psfs found at {psf_dir}"

    for psf_path in tqdm(pathlist, desc="Reading psf") if verbose else pathlist:
        psfs.append(np.array(Image.open(psf_path), dtype=float))

    if crop:
        psfs = [psf[crop[0] : crop[2], crop[1] : crop[3]] for psf in psfs]

    if patchsize:
        psfs = [diffuser_utils.pyramid_down(psf, patchsize) for psf in psfs]

    return psfs


def superimpose_psfs(psfs, blur_levels=1, one_norm=True):
    """
    Superimpose translated psf measurements of the same focus level.

    Parameters
    ----------
    psfs : list, np.ndarray
        list of 2d numpy arrays (y, x) or stacked 4d numpy array (n_blur, positions, y, x)
    blur_levels : int, optional
        number of focus levels, by default 1
    one_norm : bool, optional
        whether to one-normalize superimposed images, by default True

    Returns
    -------
    list or np.ndarray
        superimposed image or list of images for each focus level
    """
    if not isinstance(psfs, np.ndarray):
        assert len(psfs) % blur_levels == 0, "Incorred number of psfs to superimpose"
        psfs_shape = (blur_levels, len(psfs) // blur_levels) + psfs[0].shape
        psfs = np.stack(psfs).reshape(psfs_shape).transpose(1, 0, 2, 3)

    n_blur, _, y, x = psfs.shape
    supimp_imgs = np.zeros((n_blur, y, x))
    for i in range(n_blur):
        img = np.sum(psfs[i], axis=0)

        # need one-norming to go after superimposing because some psfs get cut off
        if one_norm:
            img = np.round(one_normalize(img.astype(float)) * 255).astype(np.int16)
        supimp_imgs[i] = img

    if blur_levels == 1:
        supimp_imgs[0]
    else:
        return supimp_imgs


def view_coef_psf_rings(
    coeffs,
    dim=256,
    circle_radii=[0, 50, 100, 150, 200],
    points_per_ring=24,
    device=torch.device("cpu"),
):
    """
    Return an image of circles of simulated psfs for the given coefficients at the
    specified radii

    Parameters
    ----------
    coeffs : torch.Tensor
        seidel coefficients (6,1)
    circle_radii : list, optional
        list of radii, by default [0, 50, 100]
    points_per_ring : int, optional
        points per radius, by default 24
    device : torch.device, optional
        device to compute psfs on, by default torch.device("cpu")
    """
    circ_points = []
    for r in circle_radii:
        ppr = points_per_ring
        if r == 0:
            ppr = 1
        circ_points += util.getCircList((0, 0), radius=r, num_points=ppr)

    psfs = seidel.compute_psfs(
        coeffs.to(device),
        circ_points,
        dim=dim,
        device=device,
    )
    return torch.sum(torch.stack(psfs, 0), 0).cpu().numpy()


def view_patched_psf_rings(
    psf_patches,
    psf_coordinates,
    dim=256,
    psf_dim=64,
    points_per_ring=24,
    threshold=0.7,
    center_tolerance=5,
):
    """
    Return an image of circles of rotates psf patches around their radii.

    Parameters
    ----------
    psf_patches : np.ndarray
        stack of cropped psf patches (assumed centered), (position, y, x)
    psf_coordinates : np.ndarray
        stack coordinates for each psf patch (position, 2)
    dim : int, optional
        output dimension in y and x of image (square only)
    psf_dim : int, optional
        maximum size of signal around psf coordinate
    points_per_ring : int, optional
        rotated psf examples to display per psf, by default 24
    threshold : float
        psf mask threshold (by quantile) for preprocessing
    center_tolerance : int
        tolerance (by pixel radius) to be seen as a "center" psf

    Returns
    -------
    np.ndarray
        summed rings of rotated psfs
    """

    centered_coords = psf_coordinates - (dim // 2)
    radii = [diffuser_utils.get_radius(c[0], c[1]) for c in centered_coords]

    circ_points = []
    for r in radii:
        ppr = points_per_ring
        if r < center_tolerance:
            ppr = 1
            r = 0
        circ_points.append(util.getCircList((0, 0), radius=r, num_points=ppr))

    rotated_psfs = []
    for i in tqdm(range(len(circ_points)), desc="Rendering rings"):
        for j, point in enumerate(circ_points[i]):
            rotated_psfs.append(
                rotate_psf(
                    process_psf_patch(psf_patches[i], psf_coordinates[i], threshold, psf_dim),
                    centered_coords[i],
                    point,
                    dim,
                )
            )

    rotated_psfs = np.sum(np.stack(rotated_psfs, 0), 0)

    return rotated_psfs


def plot_psf_rings(psfs, coords, blur_levels, dim, psf_dim):
    """
    Generates a plot of psf rings for the given psf set

    Parameters
    ----------
    psfs : np.ndarray
        4d numpy array of images (n_blur, positions, y, x)
    coords : np.ndarray
        3d numpy array of stacks of coordinates (n_blur, positions, 2)
    blur_levels : int
        number of blur levels
    dim : int
        size of output plot (square)
    """

    fig, ax = plt.subplots(1, blur_levels, figsize=(blur_levels * 12, 10), dpi=300)
    for i in range(blur_levels):
        rotated_psfs = view_patched_psf_rings(
            psfs[i],
            coords[i],
            dim=dim,
            psf_dim=psf_dim,
        )

        img = ax[i].imshow(rotated_psfs, cmap="gray", interpolation="none")
        fig.colorbar(img, ax=ax[i], fraction=0.046, pad=0.04)
        ax[i].set_title(f"Focus level: {i}")
        ax[i].axis("off")

    plt.suptitle("Sampled PSF rings for each focus level", fontsize=18)
    plt.show()


##### ---------- alignment axis calibration ---------- #####


def draw_dot(img_ar, coord, size, val=None):
    if val == None:
        val = np.max(img_ar)
    r = size // 2
    s = img_ar.shape
    for i in range(-r, r + 1):
        for j in range(-r, r):
            if i + coord[0] > s[0] or i + coord[0] < 0:
                continue
            if j + coord[1] > s[1] or j + coord[1] < 0:
                continue
            if i**2 + j**2 > r**2:
                continue
            img_ar[i + coord[0], j + coord[1]] = val


def find_line_intersection(p1, v1, p2, v2):
    cross_product = np.cross(v1, v2)
    if np.allclose(cross_product, [0, 0, 0]):
        return None

    t1 = np.cross(p2 - p1, v2) / cross_product
    intersection_point = np.round(p1 + t1 * v1).astype(int)
    return intersection_point


def find_all_intersections(points, vectors):
    intersections = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, v1 = points[i], vectors[i]
            p2, v2 = points[j], vectors[j]
            intersection = find_line_intersection(p1, v1, p2, v2)

            # ignore parallel shifts
            if intersection is None:
                continue
            intersections.append(intersection)
    return intersections


def estimate_alignment_center(
    psfs_path,
    blur_levels,
    anchor_foc_idx=0,
    vector_foc_idx=1,
    coord_method="peaks",
    estimate_method="median",
    conv_kern_sizes=[7, 21, 45],
    crop=None,
    verbose=True,
    plot=True,
):
    """
    Estimates the alignment center of a distortion-aberated LRI system using psf translation
    at multiple focus levels. The power set of the intersections of translation vectors provides
    a distribution for the center of the system. Estimates the center of this distribution.

    Parameters
    ----------
    psfs_path : str
        path to folder containing ordered psf measurements as .bin files
    blur_levels : int
        number of focus levels - each point should be measured blur_levels times in order
    anchor_foc_idx : int
        index of the anchor focus level, by default 0
    vector_foc_idx : int
        index of the vector focus level (used to estimate delta from anchor), by default 1
    coord_method : str
        method to find psf locations (convolution or corner_peaks), one of {'conv', 'peaks'}"
    estimate : str
        method to estimate distribution center, one of {"mean", "median"}
    crop : tuple(int, int, int, int)
        indicates a crop region (to speed up computation), (y1,x1,y2,x2)

    Returns
    -------
    int
        estimate coordinates for system alignment axis
    """
    assert blur_levels >= 2, "Must provide at least two levels of focus per point"
    assert estimate_method in ["mean", "median"], "Estimate not in {'mean', 'median'}"
    assert coord_method in ["conv", "peaks"], "coord_method not in {'conv', 'peaks'}"

    # read an preprocess measurements
    psfs = read_psfs(psfs_path, crop=crop)
    if verbose:
        print(f"Found {len(psfs)} psf measurements of shape {psfs[0].shape}.")

    # get coordinates of points at each focus level
    supimp_psfs = superimpose_psfs(psfs, blur_levels)
    if coord_method == "conv":
        coords = get_psf_coords(psfs, blur_levels, coord_method, conv_kern_sizes)
    else:
        coords = get_psf_coords(supimp_psfs, blur_levels, coord_method)

    # get anchor points and vectors
    get_vec = lambda ini, fin: np.array([fin[0] - ini[0], fin[1] - ini[1]])
    get_np_pt = lambda x: np.array(list(x))
    anchor_coords, vec_coords = coords[anchor_foc_idx], coords[vector_foc_idx]

    anchors = [get_np_pt(anchor_coords[j]) for j in range(len(coords[0]))]
    vectors = [get_vec(anchor_coords[j], vec_coords[j]) for j in range(len(coords[0]))]

    # find intersections and estimate center
    intersections = np.stack(find_all_intersections(anchors, vectors), 0)
    if estimate_method == "mean":
        center = np.round(np.mean(intersections, 0)).astype(int)
    else:
        center = np.round(np.median(intersections, 0), 0).astype(int)

    true_center = center
    if crop is not None:
        true_center = np.array([center[0] + crop[0], center[1] + crop[1]])
    if verbose:
        print(f"Estimated center: {true_center}")
    if plot:
        # draw intersections
        im1 = supimp_psfs[vector_foc_idx].copy()
        for coord in intersections:
            draw_dot(im1, coord, 5)
        draw_dot(im1, center, 15)

        # draw found coordinates
        im2 = supimp_psfs[anchor_foc_idx] + supimp_psfs[vector_foc_idx]
        for coord in anchor_coords:
            draw_dot(im2, coord, 7)
        for coord in vec_coords:
            draw_dot(im2, coord, 7)

        # plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 20))
        ax[0].imshow(im1, cmap="inferno")
        ax[0].set_title("Small: intersection coordinates, Large: center estimate")
        ax[1].imshow(im2, cmap="inferno")
        ax[1].set_title("Positions of located coordinates")
        plt.tight_layout()
        plt.show()

    return true_center


##### ---------- LRI psf simulation ---------- #####


def radial_subdivide(psf_coords, sys_center, maxval, return_radii=False):
    """
    Return a list of radial subdivision boundaries for each psf focus level given in
    psf_coords. Subdivisions are halfway between psf radii.

    Parameters
    ----------
    psf_coords : list
        list of list of coordinates for psfs at each focus level
    sys_center : tuple(int, int)
        estimate for center of system (to calculate radii from  )
    maxval : float
        maximum radial value
    return_radii : False
        whether to also return radii

    Returns
    -------
    list
        list of lists of radial subdivision boundaries
    """
    subdivisions = []
    radii = []

    # get radii and subdivide
    for i in range(len(psf_coords)):
        radii_i = sorted(
            [
                min(
                    diffuser_utils.get_radius(coord[0] - sys_center[0], coord[1] - sys_center[1]),
                    maxval,
                )
                for coord in psf_coords[i]
            ]
        )
        subdivisions_i = [(radii_i[j] + radii_i[j + 1]) // 2 for j in range(len(radii_i) - 1)]

        subdivisions.append(subdivisions_i)
        radii.append(sorted(radii_i))

    if return_radii:
        return subdivisions, radii
    return subdivisions


def plot_subdivisions(blur_levels, subdivisions, radii, superimposed_psfs):
    """
    Plots radial subdivisions found at various focus levels

    Parameters
    ----------
    blur_levels : int
        number of blur levels
    subdivisions : list
        list of lists of subdivisions for each blur level
    radii : list
        list of lists of radii for psfs at each blur level
    superimposed_psfs : np.ndarray
        3d numpy array of psfs, superimposed with each position (n_blur, y, x)
    """
    fig, ax = plt.subplots(2, blur_levels, figsize=(12 * blur_levels, 20), dpi=200)
    fig.set_dpi(70)

    for i in range(blur_levels):
        img = ax[0][i].imshow(superimposed_psfs[i], cmap="grey")
        fig.colorbar(img, ax=ax[0][i], fraction=0.046, pad=0.04)
        ax[0][i].set_title(f"Superimposed Measurements: blur {i}")
        ax[0][i].axis("off")

        ax[1][i].scatter(radii[i], range(len(radii[i])))
        ax[1][i].set_xlabel("radius")
        ax[1][i].set_ylabel("order")
        ax[1][i].set_xlim(np.array(radii).min() - 10, np.array(radii).max() + 10)
        ax[1][i].set_title(f"Subdivisions & radii: blur {i}")

        ax[1][i].vlines(
            subdivisions[i],
            ymin=0,
            ymax=len(subdivisions[i]),
            colors=["red"] * len(subdivisions),
        )
    plt.suptitle("PSF distances from alignment axis and chosen radial subdivisions", fontsize=18)
    plt.show()


def interp_psf_ramp(psfs, psf_coords, ramp_radii, dim, verbose=False):
    """
    Given psfs for different sample positions with varying blur levels
    from an LRI system and a "dim" number (number of psfs), interpolates between
    psfs at different positions of each blur level and returns a psf "ramp".

    Note: current assumption is that the psfs provided are sampled from at least
    the edges of the ramp radii. If this is nto true, psfs at the outer edges of the
    radii may not be accurate.

    Parameters
    ----------
    psfs : np.ndarray or torch.Tensor
        4d stack of centered psf patches (blur level, position, n, n)
    coords : np.ndarray or torch.Tensor
        3d stack of origin-centric coordinates for each psf patch (blur, position, 2)
    ramp_radii : list
        list of radii to return psfs for on ramp.
    dim : int
        dimension of image to use psf ramp in

    Returns
    -------
    np.ndarray or torch.Tensor
        4d stack of centered and interpolated patches (blur_level, ramp_radii, n, n)
    np.ndarray or torch.Tensor
        3d stack of psfs corresponding to the output psfs (blur_level, ramp_radii, 2)
    """
    istensor = isinstance(psfs, torch.Tensor)
    n, p, y, psf_dim = psfs.shape
    max_radius = diffuser_utils.get_radius(dim // 2, dim // 2)

    assert y == psf_dim, "psf patches must be square"
    assert max(ramp_radii) <= max_radius, "Radii out of bounds"
    if isinstance(psf_coords, torch.Tensor):
        psf_coords = psf_coords.cpu()

    device = psfs.device if istensor else torch.device("cpu")

    def tt(tens):
        if isinstance(tens, torch.Tensor):
            return tens
        return torch.tensor(tens, device=device)

    # Align every psf patch to (1, 1) * k for k in radii of coordinates
    cur_radii = np.zeros((n, p))
    new_psfs = torch.zeros((n, p, psf_dim, psf_dim), device=device)

    for i in tqdm(list(range(n)), "Aligning psfs") if verbose else range(n):
        for j in range(p):
            cur_radius = diffuser_utils.get_radius(*psf_coords[i, j])
            end_pos, cur_radii[i, j] = np.ones(2) * cur_radius.item(), cur_radius
            new_psfs[i, j] = rotate_psf(tt(psfs[i, j]), psf_coords[i, j], end_pos, psf_dim, True)

    # Resample psf patches at each ramp radius via interpolating between existing psfs
    ramp_radii = np.tile(np.expand_dims(np.array(ramp_radii), 0), reps=(n, 1))
    new_coords = np.expand_dims(np.array([np.cos(45), np.sin(45)]), (0, 1))

    new_psfs = diffuser_utils.img_interp1d(
        new_psfs.cpu().numpy(), cur_radii, ramp_radii, verbose=verbose
    )
    new_coords = np.tile(new_coords, reps=(n, 1, 1)) * ramp_radii[..., None]

    if istensor:
        new_psfs, new_coords = tt(new_psfs), tt(new_coords).cpu()

    return new_psfs, new_coords


def rotate_psf(psf, source_pos, end_pos, dim, return_centered=False):
    """
    Given an (assumed centered) psf patch, computes a rotated psf image of
    dimension dim.

    Parameters
    ----------
    psf : np.ndarray or torch.Tensor
        assumede centered psf patch (y, x)
    source_pos : tuple(int, int)
        original position of psf (relative to system center)
    end_pos : tuple(int, int)
        final position of psf (relative to system center)
    dim : int
        output dimension in y and x of image (square only)
    return_centered : bool, optional
        whether to return psf rotated but just centered at the origin

    Returns
    -------
    np.ndarray or torch.Tensor
        image of size (dim,dim) containing rotated psf
    """
    # pad psf to avoid cutting off in rotation
    psf_s = psf.shape
    psf = center_pad_to_shape(psf[None, ...], (psf_s[0] * 2, psf_s[1] * 2))[0]

    # calculate the rotation angle if the patch makes a polar translation
    if np.any(np.array(source_pos) != np.array(end_pos)):
        get_rad = lambda x: np.arctan2(x[0], x[1])
        theta = get_rad(source_pos) - get_rad(end_pos)
        if isinstance(psf, torch.Tensor):
            rot_psf = vision_F.rotate(
                psf[None, ...],
                np.degrees(theta).item(),
                interpolation=vision_F.InterpolationMode.BILINEAR,
                fill=0,
            )[0]
        else:
            rot_psf = ndimage.rotate(psf, np.degrees(theta), reshape=False, cval=0)
    else:
        rot_psf = psf

    # pad rotated patch onto output shape
    if rot_psf.shape[-1] < dim:
        out_img = center_pad_to_shape(rot_psf[None, ...], (dim, dim))[0]
    else:
        out_img = center_crop_to_shape(rot_psf[None, ...], (dim, dim))[0]

    # shift into place
    if return_centered:
        end_pos = (0, 0)

    if isinstance(out_img, torch.Tensor):
        out_img = util.shift_torch(out_img, end_pos, mode="bicubic")
        return out_img
    else:
        out_img = util.shift_torch(torch.tensor(out_img), end_pos, mode="bicubic").cpu().numpy()
        return out_img


def get_psf_coords(
    psfs,
    blur_levels,
    method="conv",
    ksizes=[7, 21, 45],
    threshold=0.7,
    min_distance=12,
    verbose=False,
):
    """
    Compute the coordinate locations of each psf in the provided psfs and return
    grouped by focus level

    Parameters
    ----------
    psfs : list(np.ndarray) or np.ndarray
        list (or stack) of superimposed psf images (one / focus level) (n_blur, y, x)
    blur_levels : int
        number of different focus levels (if using a list of psfs)
    method : str, optional
        whether to use peaks or convolution for point localization, by default "conv"
    ksizes : list, optional
        list of convolutional kernel sizes, by default [7, 21, 45]
    threshold : float
        threshold for corner peaks thresholding
    min_distance : float
        min distance between two psfs (if method == "peaks")


    Returns
    -------
    list
        list of lists containing psf locations for each focus level
    """
    enum_list = lambda x: list(enumerate(x))
    coords = [[] for i in range(blur_levels)]

    if method == "conv":
        for i, psf in (
            tqdm(enum_list(psfs), desc="Centering", file=sys.stdout) if verbose else enum_list(psfs)
        ):
            psf = psf.copy()
            psf[psf < np.quantile(psf, threshold)] = 0
            ks = ksizes[i % blur_levels]
            coords[i % blur_levels].append(get_psf_center(psf, ks * 2 + 4, kernel_size=ks))
    elif method == "peaks":
        for f in (
            tqdm(range(blur_levels), desc="Centering", file=sys.stdout)
            if verbose
            else range(blur_levels)
        ):
            calib_image = psfs[f].copy()
            calib_image[calib_image < np.quantile(calib_image, threshold)] = 0

            # locate local intensity peaks
            raw_coord = feature.corner_peaks(
                morphology.erosion(psf, morphology.disk(2)),
                min_distance=min_distance,
                indices=True,
                threshold_rel=0.5,
            )
            coords[f] += raw_coord.tolist()

    return coords


### ---------- Single-API-call functions ---------- ###


def get_lsi_psfs(
    psf_dir: str,
    blur_levels: int,
    crop_size: int,
    dim: int,
    ksizes=[],
    use_first=True,
    exposures=[],
    threshold=0.7,
    zero_outside=0,
    start_idx=0,
    blurstride=1,
    verbose=True,
    norm="",
):
    """
    Reads in psfs taken at different focus levels on the alignment axis
    of an LSI system, processes them, and returns them as a stack

    psf measurements are assumed to be in ".bmp or .tiff" files, taken at every blur
    level in increasing blur order. For example, given blur_levels = 3 and the files:
        |- psf_dir
            |- 1.bmp
            |- 2.bmp
            |- 3.bmp
            |- 4.bmp
            |- 5.bmp
            |- 6.bmp
    This function assumes that 1.bmp is more in focus than 2.bmp, and will return a stack
    containing the processed [1.bmp, 2.bmp, 3.bmp].

    Parameters
    ----------
    psf_dir : str
        directory containing list of ".bmp or .tiff" psf measurements
    blur_levels : int
        number of focus levels to return (regardless of how many are in dir)
    crop_size : tuple or int
        size of crop, if int assumes square
    dim : tuple
        size of output image (will be downsampled from crop)
    ksizes : list, optional
        list of kernel sizes (for conv coord finding),
        by default [7,21,45,55,65]
    use_first : bool, optional
        Whether to use the first blur center as the center for each blur
        level, or to attempt to find a center for each level. Often the
        first center is the best, assuming no optical distortion, by
        default True
    exposures : list, optional
        list of floats representing exposure lengths for eahc blur level,
        by default []
    threshold : float, optional
        quantile threshold value for noise removal, by default 0.7
    zero_outside : int, optional
        radius to zero psf outside of, used for pixel noise, by default 0
    start_idx : int, optional
        most in-focus index, by default 0
    blurstride : int, optional
        stride along (assumed in order) blur levels, by default 1
    norm : str, optional
        which norm to divide psfs by (l1, l2), one of {"", "one", "two"},
        by default ""
    Returns
    -------
    np.ndarray
        stack of centered, processed psf measurements (n_blur, y, x)
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    ################ Read in psfs #################
    psfs = read_psfs(psf_dir, verbose=verbose)
    if blurstride < 0:  # only defocused psfs
        psfs = psfs[-1:]
    else:
        psfs = psfs[start_idx : blur_levels * blurstride : blurstride]

    ################ Locate psf centers #################
    centers = []
    if use_first:
        first_ksize = ksizes[0] if len(ksizes) > 0 else 7
        first_center = get_psf_center(psfs[0], kernel_size=first_ksize)
        centers = [first_center] * len(psfs)
    else:
        for i, psf in tqdm(list(enumerate(psfs)), "Centering", file=sys.stdout):
            ksize = ksizes[i] if i < len(ksizes) else 7
            centers.append(get_psf_center(psf, kernel_size=ksize))

    ################ Process patches (centering, noise reduction, thresholding) #################
    for i in range(len(centers)):
        psfs[i] = process_psf_patch(psfs[i], centers[i], threshold, crop_size)

    if exposures:
        psfs = even_exposures(psfs, blur_levels, exposures, verbose=verbose)

    if zero_outside:
        psfs = [apply_mask_center(psf, zero_outside * 2, "circle") for psf in psfs]

    ################ Downsample patches #################
    if crop_size[0] != dim[0] or crop_size[1] != dim[1]:
        assert crop_size[0] > dim[0] or crop_size[1] > dim[1], "Patch upsampling is not supported"
        psfs = [diffuser_utils.pyramid_down(psf, dim) for psf in psfs]

    ############### Apply specified norm #################
    if norm:
        fn = np.sum if norm == "one" else np.linalg.norm
        psfs = [psf / fn(psf) for psf in psfs]

    return np.stack(psfs)


def get_lri_psfs(
    psf_dir: np.ndarray,
    blur_levels: int,
    crop_size: int,
    dim: int,
    alignment_estimate=None,
    coord_method="conv",
    ksizes=[7, 21, 45, 55, 65],
    min_distance=12,
    exposures=None,
    threshold=0.7,
    psf_dim=120,
    polar=True,
    use_psf_ramp=True,
    device=torch.device("cpu"),
    verbose=False,
    plot=False,
):
    """
    Reads in psf measurements taken at different focus levels and radii
    from a directory, processes them, and returns a psf_data tensor,
    containing LRI psfs in polar form for each ring.

    psf measurements are assumed to be in ".bmp or .tiff" files, taken at every blur
    level at each position before moving on to the next. For example, given
    blur_levels = 3 and the files:
        |- psf_dir
            |- 1.bmp
            |- 2.bmp
            |- 3.bmp
            |- 4.bmp
            |- 5.bmp
            |- 6.bmp
    1, 2, and 3 are assumed to be at the same position, (1,4), (2,5), (3,6) at
    the same level of blur.

    Note: if interp_between_pos is true, this method will attempt to create a
    smooth set of psfs via interpolation. This is extremely important to avoid
    circular artifacts.

    Parameters
    ----------
    psf_dir : str
        directory containing list of ".bmp or .tiff" psf measurements
    blur_levels : int
        number of focus levels per position
    crop_size : int
        size of crop (square)
    dim : int
        size of output image (will be downsampled from crop) (square)
    alignment_estimate : tuple(int, int), optional
        coordinates of axis of alignment in measurements, if none given will
        use the center of the first psf measurement
    coord_method : str, optional
        method for finding psf centers, one of {conv, peaks}, by default "conv"
    ksizes : list, optional
        list of kernel sizes (for conv coord finding), by default [7,21,45,55,65]
    min_distance : int, optional
        minimum distance between two psfs (for peaks coord finding), by default 12
    exposures : list, optional
        list of floats of exposure length for each focus level, by default None
    psf_dim : int, optional
        size of cropping around each psf (should be > largest psf after downsampling),
        by default 120
    threshold : float, optional
        quantile threshold value for psf processing, by default 0.7
    polar : bool, optional
        whether to return the polar (RoFT) or real space psf stack
    use_psf_ramp : bool, optional
        whether to use an interpolated ramp of psfs between positions, by default True
    device : torch.Device
        device to run psf rendering on
    """

    ########### Subroutine definitions and safety checks ############
    def process_patches(psf_stack, coord_stack):
        procd_patches = np.zeros(psf_stack.shape[:2] + (psf_dim, psf_dim))
        for i in range(psf_stack.shape[0]):
            for j in range(psf_stack.shape[1]):
                patch, coord = psf_stack[i, j], coord_stack[i, j] + dim // 2
                procd_patches[i, j] = process_psf_patch(patch, coord, threshold, psf_dim)

        return procd_patches

    assert psf_dim < dim, "size of psf must be smaller than image size"
    assert crop_size >= dim, "crop must be greater or equal to final size"

    ############### Read psfs with crop to speed up centering ###############
    image_crop = (
        alignment_estimate[0] - crop_size // 2,
        alignment_estimate[1] - crop_size // 2,
        alignment_estimate[0] + crop_size // 2,
        alignment_estimate[1] + crop_size // 2,
    )
    psfs = read_psfs(psf_dir, crop=image_crop, patchsize=(dim, dim), verbose=verbose)
    assert len(psfs) % blur_levels == 0, "Requires same number of psfs for each pos"
    for i, im in enumerate(psfs):
        dynamic_range = np.max(im) - np.min(im)
        if dynamic_range <= np.max(im - np.min(im)) * 0.1:
            raise AssertionError(
                f"Found psf {i} with insufficient dynamic range, "
                "likely due to no psf being in crop. Consider increasing 'crop_size'."
            )

    if exposures is not None:
        psfs = even_exposures(psfs, blur_levels, exposures, verbose=verbose)

    coords = np.array(
        get_psf_coords(
            psfs,
            blur_levels,
            method=coord_method,
            ksizes=ksizes,
            threshold=threshold,
            min_distance=min_distance,
            verbose=verbose,
        )
    )

    ############ Process psfs and coordinates #############
    psfs_shape = (blur_levels, len(psfs) // blur_levels) + (dim, dim)
    radius_order = [diffuser_utils.get_radius(*coord) for coord in coords[0] - dim // 2]

    coords = coords[:, np.argsort(radius_order), ...]
    procd_coords = coords - (dim // 2)

    psfs = np.stack(psfs).reshape(psfs_shape).transpose(1, 0, 2, 3)
    procd_psfs = psfs[:, np.argsort(radius_order), ...]
    procd_psfs = process_patches(procd_psfs, procd_coords)

    procd_psfs = torch.tensor(procd_psfs, device=device)
    procd_coords = torch.tensor(procd_coords)

    ############### Getting radial image subdivisions and psf ramp ################
    maxval = diffuser_utils.get_radius(dim // 2, dim // 2)
    subdivisions, radii = radial_subdivide(coords, (dim // 2, dim // 2), maxval, True)

    rs = np.linspace(0, (dim / 2), dim, endpoint=False, retstep=False)
    point_list = [(r, -r) for r in rs]  # radial line of PSFs

    if plot:
        supimp_psfs = superimpose_psfs(psfs, blur_levels, one_norm=False)
        plot_subdivisions(blur_levels, subdivisions, radii, supimp_psfs)

    if use_psf_ramp:
        ramp_radii = [diffuser_utils.get_radius(*r) for r in point_list]
        procd_psfs, procd_coords = interp_psf_ramp(
            procd_psfs, procd_coords, ramp_radii, dim, verbose=verbose
        )

    ############### Computing psf data ################
    psf_data = torch.zeros((blur_levels, dim, len(point_list), dim), device=device)
    for n in list(range(blur_levels)):
        print(f"Rendering {n + 1}/{blur_levels}:")
        for theta in tqdm(list(range(dim)), file=sys.stdout) if verbose else list(range(dim)):
            if use_psf_ramp:
                psf_idx = theta
            else:
                current_radius = diffuser_utils.get_radius(*point_list[theta])
                psf_idx = sum([1 if current_radius > div else 0 for div in subdivisions[0]])
            psf_data[n, theta] = rotate_psf(
                psf=procd_psfs[n][psf_idx],
                source_pos=procd_coords[n][psf_idx],
                end_pos=(-point_list[theta][1], point_list[theta][0]),
                dim=dim,
            )

    if plot:
        plot_psf_rings(psfs, coords, blur_levels, dim, psf_dim)

    ############### Computing RoFT of psf data ################
    if polar:
        psf_data = polar_transform.batchimg2polar(psf_data, numRadii=len(point_list))
        n, z, t, a = psf_data.shape
        psf_data = torch.concat((psf_data, torch.zeros((n, z, 2, a), device=device)), dim=2)

        # Not batching because of memory constraints with fft.rfft
        for i in tqdm(list(range(n)), desc="Computing RoFTs"):
            temp_rft = torch.fft.rfft(psf_data[i, :, 0:-2, :], dim=1)
            psf_data[i, :, 0 : psf_data.shape[2] // 2, :] = torch.real(temp_rft)
            psf_data[i, :, psf_data.shape[2] // 2 :, :] = torch.imag(temp_rft)
            del temp_rft

        gc.collect()
        torch.cuda.empty_cache()

        # add together the real and imaginary parts of the RoFTs due to symmetry
        psf_data = (
            (
                psf_data[:, :, 0 : psf_data.shape[2] // 2, :]
                + 1j * psf_data[:, :, psf_data.shape[2] // 2 :, :]
            )
            .cpu()
            .numpy()
        )
        gc.collect()
        torch.cuda.empty_cache()
        return psf_data

    return psf_data.cpu().numpy()


def save_lri_psf(coeffs, psf_data, psf_dir, focus_level):
    coeffs, psf_data = coeffs.detach().cpu().numpy(), psf_data.detach().cpu().numpy()
    save_data = {"coefficients": coeffs, "psf_data": psf_data}
    if not os.path.exists(psf_dir):
        os.makedirs(psf_dir)
    path = os.path.join(psf_dir, f"lri_psf_calib_{focus_level}.mat")
    io.savemat(path, save_data)
    print(f"Saved psf data of shape {psf_data.shape} to {path}.")


def load_lri_psfs(psf_dir, num_ims):
    psf_data = []
    for i in range(num_ims):
        mat = io.loadmat(os.path.join(psf_dir, f"lri_psf_calib_{i}.mat"))
        psf_data.append(mat["psf_data"])

    return np.stack(psf_data, 0)


def load_psf_npy(psf_dir, norm=None):
    files = os.listdir(psf_dir)
    # sort by number in filename
    files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[-1]))
    psfs = []
    for file in files:
        if file.endswith(".npy"):
            psf = np.load(os.path.join(psf_dir, file))
            psfs.append(psf)

    if norm:
        fn = np.sum if norm == "one" else np.linalg.norm
        psfs = [psf / fn(psf) for psf in psfs]

    return np.stack(psfs, 0)
