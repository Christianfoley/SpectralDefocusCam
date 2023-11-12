# utils for experimental psfs
import sys, glob, os, tqdm
import numpy as np
from scipy import io, ndimage, signal
import torch
import cv2
from skimage import feature, morphology
import matplotlib.pyplot as plt

from PIL import Image
import utils.diffuser_utils as diffuser_utils
from models.rdmpy._src import seidel, util


### ---------- Utility functions ---------- ##


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
        return psf * np.where(psf > np.quantile(psf, quantile), 1, 0)


def denoise_speckle(img, thresh_val=None):
    if thresh_val == None:
        thresh_val = np.median(img)

    # get binary de-speckling mask
    denoise_mask = cv2.medianBlur((img > thresh_val).astype("float32"), ksize=5)
    denoise_mask = denoise_mask > np.max(denoise_mask) / 2

    denoised_img = np.where(denoise_mask > 0, img, thresh_val)
    return denoised_img


def get_circular_kernel(diameter):
    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

    return kernel


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


def center_crop_to_shape(psfs, shape):
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


def center_crop_psf(psf, width, shape="square", center_offset=None, kernel_size=7):
    # pad to allow larger crop
    padding = (
        (psf.shape[0] // 2, psf.shape[0] // 2),
        (psf.shape[1] // 2, psf.shape[1] // 2),
    )
    psf = np.pad(psf, padding)

    if center_offset is None:
        # blur for center of mass
        psf = psf * np.where(psf > np.quantile(psf, 0.75), psf, 0)
        psf_conv = cv2.GaussianBlur(psf, (kernel_size, kernel_size), 0)
        # kernel = get_circular_kernel(kernel_size)
        # psf_conv = signal.correlate(psf, kernel, mode="same")

        max_index = np.argmax(psf_conv)
        max_coords = np.unravel_index(max_index, psf.shape)
    else:
        max_coords = (
            center_offset[0] + padding[0][0],
            center_offset[1] + padding[1][0],
        )

    # crop around max (always returns even, favors left&above)
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
        circle_crop = np.multiply(get_circular_kernel(width), square_crop)
        return circle_crop, max_coords
    else:
        raise AssertionError("unhandled crop shape")


def read_psfs(psf_dir, crop=None, patchsize=None):
    """
    Reads in ordered psf measurements stored as .bmp files from a directory.

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
    psfs = []
    for psf_path in tqdm.tqdm(pathlist, "Reading psf"):
        psfs.append(np.array(Image.open(psf_path), dtype=float))

    if crop:
        psfs = [psf[crop[0] : crop[2], crop[1] : crop[3]] for psf in psfs]

    if patchsize:
        psfs = [diffuser_utils.pyramid_down(psf, patchsize) for psf in psfs]

    return psfs


def superimpose_psfs(psf_list, focus_levels=1, one_norm=True):
    """
    Superimpose translated psf measurements of the same focus level.

    Parameters
    ----------
    psf_list : str
        directory containing ordered psf measurements
    focus_levels : int, optional
        number of focus levels, by default 1
    one_norm : bool, optional
        whether to one-normalize superimposed images, by default True

    Returns
    -------
    list or np.ndarray
        superimposed image or list of images for each focus level
    """
    supimp_imgs = []
    psf_list = [
        [psf for i, psf in enumerate(psf_list) if i % focus_levels == f]
        for f in range(focus_levels)
    ]
    for foc_psfs in psf_list:
        img = np.sum(np.stack(foc_psfs, 0), 0)

        # need one-norming to go after superimposing because some psfs get cut off
        if one_norm:
            img = np.round(one_normalize(img.astype(float)) * 255).astype(np.int16)
        supimp_imgs.append(img)

    if focus_levels == 1:
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
    psf_patches, psf_coordinates, dim=256, psf_dim=64, points_per_ring=24, threshold=0.7
):
    """
    Return an image of circles of rotates psf patches around their radii.

    Parameters
    ----------
    psf_patches : list
        list of cropped psf patches (assumed centered)
    psf_coordinates : list
        list of tuples of coordinates for each psf patch
    dim : int, optional
        output dimension in y and x of image (square only)
    psf_dim : int, optional
        maximum size of signal around psf coordinate
    points_per_ring : int, optional
        rotated psf examples to display per psf, by default 24
    threshold : float
        psf mask threshold (by quantile) for preprocessing

    Returns
    -------
    _type_
        _description_
    """

    def process_patch(patch, coords):
        patch = patch.copy()
        patch = patch[
            coords[0] - psf_dim // 2 : coords[0] + psf_dim // 2,
            coords[1] - psf_dim // 2 : coords[1] + psf_dim // 2,
        ]
        patch = thresh(one_normalize(patch), quantile=threshold)
        return patch

    centered_coords = [(c[0] - dim // 2, c[1] - dim // 2) for c in psf_coordinates]
    radii = [diffuser_utils.get_radius(*c) for c in centered_coords]

    circ_points = []
    for r in radii:
        ppr = points_per_ring
        if r == 0:
            ppr = 1
        circ_points.append(util.getCircList((0, 0), radius=r, num_points=ppr))

    rotated_psfs = np.zeros((len(circ_points) * len(circ_points[0]), dim, dim))
    for i in tqdm.tqdm(range(len(circ_points)), "Rendering rings"):
        for j, point in enumerate(circ_points[i]):
            rotated_psfs[i * len(circ_points[0]) + j] = rotate_psf(
                process_patch(psf_patches[i], psf_coordinates[i]),
                centered_coords[i],
                point,
                dim,
            )

    rotated_psfs = np.sum(rotated_psfs, 0)

    return rotated_psfs


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


def radial_subdivide(psf_coords, sys_center, maxval=None, return_radii=False):
    """
    Return a list of radial subdivision boundaries for each psf focus level given in
    psf_coords. Subdivisions are halfway between psf radii.

    Parameters
    ----------
    psf_coords : list
        list of list of coordinates for psfs at each focus level
    sys_center : tuple(int, int)
        estimate for center of system (to calculate radii from  )
    maxval : float, optional
        maximum radial value, by default None
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
        radii_i = [
            diffuser_utils.get_radius(
                coord[0] - sys_center[0], coord[1] - sys_center[1]
            )
            for coord in psf_coords[i]
        ]
        subdivisions_i = [
            (radii_i[j] + radii_i[j - 1]) // 2 for j in range(1, len(radii_i))
        ]

        subdivisions.append(subdivisions_i)
        radii.append(sorted(radii_i))

    if return_radii:
        return subdivisions, radii
    return subdivisions


def rotate_psf(psf, source_pos, end_pos, dim):
    """
    Given an (assumed centered) psf patch, computes a rotated psf image of
    dimension dim.

    Parameters
    ----------
    psf : np.ndarray
        assumede centered psf patch (y, x)
    source_pos : tuple(int, int)
        original position of psf (relative to system center)
    end_pos : tuple(int, int)
        final position of psf (relative to system center)
    dim : int
        output dimension in y and x of image (square only)

    Returns
    -------
    np.ndarray
        image of size (dim,dim) containing rotated psf
    """
    # Pad psf to avoid cutting off in rotation
    psf_s = psf.shape
    psf = center_pad_to_shape(np.expand_dims(psf, 0), (psf_s[0] * 2, psf_s[1] * 2))[0]

    # Calculate the rotation angle if the patch makes a polar translation
    if np.any(np.array(source_pos) != np.array(end_pos)):
        get_rad = lambda x: np.arctan2(x[0], x[1])
        theta = get_rad(source_pos) - get_rad(end_pos)
        rot_psf = ndimage.rotate(psf, np.degrees(theta), reshape=False)
    else:
        rot_psf = psf

    # Place rotated patch onto output img
    out_img = np.zeros((dim * 2, dim * 2))
    end_pos = (end_pos[0] + dim, end_pos[1] + dim)  # center -> corner
    out_img[
        end_pos[0] - psf.shape[0] // 2 : end_pos[0] + psf.shape[0] // 2,
        end_pos[1] - psf.shape[1] // 2 : end_pos[1] + psf.shape[1] // 2,
    ] += rot_psf
    out_img = center_crop_to_shape(np.expand_dims(out_img, 0), (dim, dim))[0]
    return out_img


def get_psf_coords(
    psfs,
    focus_levels,
    method="conv",
    ksizes=[7, 21, 45],
    threshold=0.7,
    min_distance=12,
):
    """
    Compute the coordinate locations of each psf in the provided psfs and return
    grouped by focus level

    Parameters
    ----------
    psfs : list(np.ndarray)
        list of single-psf images (many) or superimposed psf images (one / focus level)
    focus_levels : int
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
    tqdmify = lambda x: list(enumerate(x))  # ;)
    coords = [[] for i in range(focus_levels)]
    if method == "conv":
        for i, psf in tqdm.tqdm(tqdmify(psfs), desc="Centering", file=sys.stdout):
            psf = psf.copy()
            psf[psf < np.quantile(psf, threshold)] = 0
            ks = ksizes[i % focus_levels]
            coords[i % focus_levels].append(center_crop_psf(psf, 64, kernel_size=ks)[1])
    elif method == "peaks":
        for f in tqdm.tqdm(range(focus_levels), desc="Centering", file=sys.stdout):
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


def estimate_alignment_center(
    psfs_path,
    focus_levels,
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
    focus_levels : int
        number of focus levels - each point should be measured focus_levels times in order
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
    assert focus_levels >= 2, "Must provide at least two levels of focus per point"
    assert estimate_method in ["mean", "median"], "Estimate not in {'mean', 'median'}"
    assert coord_method in ["conv", "peaks"], "coord_method not in {'conv', 'peaks'}"

    # read an preprocess measurements
    psfs = read_psfs(psfs_path, crop=crop)
    if verbose:
        print(f"Found {len(psfs)} psf measurements.")

    # get coordinates of points at each focus level
    supimp_psfs = superimpose_psfs(psfs, focus_levels)
    if coord_method == "conv":
        coords = get_psf_coords(psfs, focus_levels, coord_method, conv_kern_sizes)
    else:
        coords = get_psf_coords(supimp_psfs, focus_levels, coord_method)

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


### ---------- Single-API-call functions ---------- ###


def get_psfs_dmm_37ux178(
    psf_dir,
    ext=[".bmp", ".tiff"],
    center_crop_width=128,
    center_crop_shape="square",
    kernel_sizes=None,
    usefirst=False,
    blurstride=1,
):
    """
    Utility function for getting LSI psfs captured on a DMM_37UX178 sony cmos sensor.
    Reads, locates, thresholds, and crops psf measurements. Returns as list of numpy
    arrays.

    Parameters
    ----------
    psf_dir : str, optional
        path to psf dir containing list of psf measurements at different focus levels
    ext : list, optional
        file extension of psf measurements, by default [".bmp", ".tiff"]
    center_crop_width : int, optional
        width of cropped chunk (centered on center of psf), if 0 will not crop, by default 128
    center_crop_shape : str, optional
        shape to crop, one of {'circle', 'square'}, by default "square"
    kernel_sizes : list, optional
        size of convolutional kernels used to find center of psf at each blur level
    usefirst : bool, optional
        whether to center all psfs around position of first (assumes invariant distortion),
        by default False
    blurstride : int, optional
        stride to sample blur levels, by default 1


    Returns
    -------
    list
        list of centered psf measurements as number arrays of shape (width, width)
    """
    psfs = []
    filenames = glob.glob(os.path.join(psf_dir, "*" + ext[0])) + glob.glob(
        os.path.join(psf_dir, "*" + ext[1])
    )
    filenames.sort()
    filenames = filenames[::blurstride]

    # open and convert to numpy
    psfs = [np.array(Image.open(file), dtype=float) for file in filenames]

    # if specified, crop all psfs to center of the in-focus psf
    offset = None
    if usefirst:
        _, offset = center_crop_psf(psfs[0], center_crop_width)

    # crop with offset
    if center_crop_width > 0:
        cropped = []
        for i, psf in enumerate(psfs):
            ksize = 7
            if kernel_sizes is not None:
                ksize = kernel_sizes[i]
            cropped.append(
                center_crop_psf(
                    psf, center_crop_width, center_crop_shape, offset, ksize
                )[0]
            )
        psfs = cropped

    return psfs


def get_lsi_psfs(
    psf_dir,
    num_ims,
    mask_shape,
    padded_shape,
    one_norm=True,
    threshold=0.7,
    blurstride=1,
):
    stack_xy_shape = mask_shape[-2:]
    psfs = get_psfs_dmm_37ux178(
        psf_dir,
        center_crop_width=min(stack_xy_shape),
        center_crop_shape="square",
        usefirst=True,
        blurstride=blurstride,
    )[:num_ims]

    # normalize
    if one_norm:
        psfs = [one_normalize(psf) for psf in psfs]

    # threshold
    if threshold:
        psfs = [thresh(psf, quantile=threshold) for psf in psfs]

    # center pad -> downsample to specified mask shape
    psfs = [
        np.squeeze(center_pad_to_shape(np.expand_dims(psf, 0), padded_shape))
        for psf in psfs
    ]
    psfs = np.stack(
        [diffuser_utils.pyramid_down(psf, stack_xy_shape) for psf in psfs], 0
    )
    return np.transpose(psfs, (0, 1, 2))


def save_lri_psf(coeffs, psf_data, psf_dir, focus_level):
    coeffs, psf_data = coeffs.detach().cpu().numpy(), psf_data.detach().cpu().numpy()
    save_data = {"coefficients": coeffs, "psf_data": psf_data}
    if not os.path.exists(psf_dir):
        os.makedirs(psf_dir)
    path = os.path.join(psf_dir, f"lri_psf_calib_{focus_level}.mat")
    io.savemat(path, save_data)
    print(f"Saved psf data of shape {psf_data.shape} to {path}.")


def get_lri_psfs(psf_dir, num_ims):
    psf_data = []
    for i in range(num_ims):
        mat = io.loadmat(os.path.join(psf_dir, f"lri_psf_calib_{i}.mat"))
        psf_data.append(mat["psf_data"])

    return np.stack(psf_data, 0)
