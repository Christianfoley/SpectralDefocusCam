# utils for experimental psfs
import sys, glob, os, tqdm
import numpy as np
import cv2
from skimage import feature, morphology
import scipy.signal as signal
import matplotlib.pyplot as plt

from PIL import Image
import utils.diffuser_utils as diffuser_utils


### ---------- Utility functions ---------- ##


def one_normalize(im):
    im = im - np.min(im)
    return im / np.max(im)


def thresh(psf, proportion_of_max=0.33, use_otsu=False):
    if use_otsu:
        blur = cv2.GaussianBlur(psf, (5, 5), 0)
        mask = cv2.threshold(
            blur, np.max(blur) / 3, np.max(blur), cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        return psf * mask
    else:
        return psf * (psf > (np.max(psf) * proportion_of_max))


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


def center_crop_psf(psf, width, shape="square", center_offset=None, kernel_size=7):
    # pad to allow larger crop
    padding = (
        (psf.shape[0] // 2, psf.shape[0] // 2),
        (psf.shape[1] // 2, psf.shape[1] // 2),
    )
    psf = np.pad(psf, padding)

    if center_offset is None:
        # blur for more accurate center
        psf_conv = blur = cv2.GaussianBlur(psf, (kernel_size, kernel_size), 0)
        # kernel = get_guassian_kernel(15)
        # psf_conv = signal.correlate(psf, kernel, mode="same")

        # get coords of max
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


def read_psfs(psf_dir, crop=None):
    """
    Reads in ordered psf measurements stored as .bmp files from a directory.

    Parameters
    ----------
    psf_dir : str
        directory containing measurements
    crop : tuple, optional
        cropping tuple: (y1, x1, y1, x2), by default None

    Returns
    -------
    list
        list of psf measurements as numpy arrays
    """
    pathlist = sorted(glob.glob(os.path.join(psf_dir, "*.bmp")))
    psfs = [np.array(Image.open(x), dtype=float) for x in pathlist]

    if crop:
        psfs = [psf[crop[0] : crop[2], crop[1] : crop[3]] for psf in psfs]

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
        if one_norm:
            img = one_normalize(img)
            img = np.round(img * 255).astype(np.int16)
        supimp_imgs.append(img)

    if focus_levels == 1:
        supimp_imgs[0]
    else:
        return supimp_imgs


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


def get_point_coords(calib_image, min_dist=30, threshold=0.5):
    """Credit to https://github.com/apsk14/rdmpy"""
    psf = calib_image.copy()
    psf[psf < 0] = 0
    psf[psf < np.quantile(psf, 0.9)] = 0

    # locate local intensity peaks
    raw_coord = feature.corner_peaks(
        morphology.erosion(psf, morphology.disk(2)),
        min_distance=min_dist,
        indices=True,
        threshold_rel=threshold,
    )
    return raw_coord.tolist()  # coord_list


def estimate_alignment_center(
    psfs_path,
    focus_levels,
    anchor_foc_idx=0,
    vector_foc_idx=1,
    coord_method="peaks",
    estimate_method="median",
    conv_kern_sizes=[7, 15, 21],
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
    psfs = list(enumerate(read_psfs(psfs_path, crop=crop)))
    if verbose:
        print(f"Found {len(psfs)} psf measurements.")

    # get coordinates of points at each focus level
    supimp_psfs = superimpose_psfs([tup[1] for tup in psfs], focus_levels)
    coords = [[] for i in range(focus_levels)]
    if coord_method == "conv":
        for i, psf in tqdm.tqdm(psfs, desc="Get centers", file=sys.stdout):
            ks = conv_kern_sizes[i % focus_levels]
            coords[i % focus_levels].append(center_crop_psf(psf, 64, kernel_size=ks)[1])
    elif coord_method == "peaks":
        for f in tqdm.tqdm(range(focus_levels), desc="Get centers", file=sys.stdout):
            coords[f] += get_point_coords(supimp_psfs[f])

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
    psf_dir="/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/psfs_10_18_2023_new_noiseavg32",
    ext=[".bmp", ".tiff"],
    scale=False,
    threshold=False,
    center_crop_width=128,
    center_crop_shape="square",
    usefirst=False,
    blurstride=1,
):
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
        psfs = [
            center_crop_psf(psf, center_crop_width, center_crop_shape, offset)[0]
            for psf in psfs
        ]
    if scale:
        psfs = [psf / 255 for psf in psfs]

    if threshold:
        psfs = [thresh(psf) for psf in psfs]

    return psfs


def get_psf_stack(
    psf_dir, num_ims, mask_shape, padded_shape, one_norm=True, blurstride=1
):
    stack_xy_shape = mask_shape[1:]
    psfs = get_psfs_dmm_37ux178(
        psf_dir,
        center_crop_width=min(stack_xy_shape),
        center_crop_shape="square",
        usefirst=True,
        blurstride=blurstride,
    )

    # normalize
    if one_norm:
        psfs = psfs[0:num_ims]
        masks = [np.where(sel > np.median(sel), 1, 0) for sel in psfs]
        psfs = [one_normalize(sel * masks[i]) for i, sel in enumerate(psfs)]

    # center pad -> downsample to specified mask shape
    psfs = [
        np.squeeze(center_pad_to_shape(np.expand_dims(psf, 0), padded_shape))
        for psf in psfs
    ]
    psfs = np.stack(
        [diffuser_utils.pyramid_down(psf, stack_xy_shape) for psf in psfs], 0
    )
    return np.transpose(psfs, (0, 1, 2))
