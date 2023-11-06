import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io
import yaml
from IPython.core.display import display, HTML
from ipywidgets import interact, widgets, fixed

import sys

sys.path.append("spectral_diffusercam_utils/")


def get_now():
    """Returns current time in YYYY_MM_DD_HH_SS"""
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")


def get_device(device):
    """
    Returns a string representing the device to be used for PyTorch computations.
    Raises:
        ValueError: If "gpu" is specified as the input argument and no GPUs are available.

    :params device (str): A string indicating whether to use "cpu" or "gpu".

    :return str: If "cpu" is specified as the input argument, returns "cpu".
             Otherwise, returns a string of the form "cuda:K", where K is the device number
             of the GPU with the most available memory at the time of execution.
    """
    if device == "cpu":
        return "cpu"
    elif isinstance(device, int):
        return f"cuda:{device}"
    else:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise ValueError("No GPU info visible from this environment")
        else:
            max_memory = 0
            max_device = 0
            for i in range(device_count):
                memory = torch.cuda.max_memory_allocated(i)
                if memory > max_memory:
                    max_memory = memory
                    max_device = i
            return f"cuda:{max_device}"


def write_yaml(yml_dict, yml_filename):
    """
    Writes dict as yml file.

    :param dict yml_dict: Dictionary to be written
    :param str yml_filename: Full path file name of yml
    """
    file_string = yaml.safe_dump(yml_dict)
    with open(yml_filename, "w") as f:
        f.write(file_string)


def read_config(config_fname):
    """Read the config file in yml format

    :param str config_fname: fname of config yaml with its full path
    :return: dict config: Configuration parameters
    """

    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    config["config_fname"] = config_fname

    return config


def plotf2(r, img, ttl, sz):
    # fig = plt.figure(figsize=(2, 2));
    # plt.figure(figsize=(20, 20));
    plt.title(ttl + " {}".format(r))
    plt.imshow(img[:, :, r], cmap="gray", vmin=0, vmax=np.max(img))
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(sz)
    fig.set_dpi(100)
    plt.show()
    # display(fig)
    # clear_output(wait=True)
    return


def plt3D(img, title="", size=(5, 5)):
    # fig = plt.figure(figsize=sz);
    interact(
        plotf2,
        r=widgets.IntSlider(min=0, max=np.shape(img)[-1] - 1, step=1, value=1),
        img=fixed(img),
        continuous_update=False,
        ttl=fixed(title),
        sz=fixed(size),
    )


def value_norm(x):
    x = x - np.min(x)
    return x / np.max(x)


def crop(x):
    DIMS0 = x.shape[0] // 2  # Image Dimensions
    DIMS1 = x.shape[1] // 2  # Image Dimensions

    PAD_SIZE0 = int((DIMS0) // 2)  # Pad size
    PAD_SIZE1 = int((DIMS1) // 2)  # Pad size

    C01 = PAD_SIZE0
    C02 = PAD_SIZE0 + DIMS0  # Crop indices
    C11 = PAD_SIZE1
    C12 = PAD_SIZE1 + DIMS1  # Crop indices
    return x[C01:C02, C11:C12, :]


def pre_plot(x, flip=True):
    if flip:
        x = np.fliplr(np.flipud(x))
    x = x / np.max(x)
    x = np.clip(x, 0, 1)
    return x


def stack_rgb_opt_30(
    reflArray,
    channels=30,
    offset=0,
    opt=None,
    scaling=[1, 1, 2.5],
):
    color_dict = scipy.io.loadmat(opt)
    red = color_dict["red"]
    green = color_dict["green"]
    blue = color_dict["blue"]

    reflArray = reflArray / np.max(reflArray)

    red_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    green_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    blue_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))

    for i in range(0, channels):
        ndx = int(offset + (i * 64 / channels) // 1)
        red_channel = red_channel + reflArray[:, :, i] * red[0, ndx] * scaling[0]
        green_channel = green_channel + reflArray[:, :, i] * green[0, ndx] * scaling[1]
        blue_channel = blue_channel + reflArray[:, :, i] * blue[0, ndx] * scaling[2]

    red_channel = red_channel / channels
    green_channel = green_channel / channels
    blue_channel = blue_channel / channels

    stackedRGB = np.stack((red_channel, green_channel, blue_channel), axis=2)

    return stackedRGB  # original version for 64 channels


def stack_rgb_opt(
    reflArray,
    opt="utils/false_color_calib.mat",
    scaling=[1, 1, 2.5],
):
    if not opt:
        opt = "utils/false_color_calib.mat"
    color_dict = scipy.io.loadmat(opt)
    red = color_dict["red"]
    green = color_dict["green"]
    blue = color_dict["blue"]

    reflArray = reflArray / np.max(reflArray)

    red_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    green_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    blue_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))

    for i in range(0, 64):
        red_channel = red_channel + reflArray[:, :, i] * red[0, i] * scaling[0]
        green_channel = green_channel + reflArray[:, :, i] * green[0, i] * scaling[1]
        blue_channel = blue_channel + reflArray[:, :, i] * blue[0, i] * scaling[2]

    red_channel = red_channel / 64.0
    green_channel = green_channel / 64.0
    blue_channel = blue_channel / 64.0

    stackedRGB = np.stack((red_channel, green_channel, blue_channel), axis=2)

    return stackedRGB


def preprocess(mask, psf, im):
    # Crop indices
    c1 = 100
    c2 = 420
    c3 = 80
    c4 = 540  # indices for 64 channel image
    # c1 = 260-128; c2 = c1 + 256; c3 = 310 - 128; c4 = c3 + 256

    # Crop and normalize mask
    mask = mask[c1:c2, c3:c4, :]
    mask = mask / np.max(mask)

    # Crop and normalize PSF
    psf = psf[c1:c2, c3:c4]
    psf = psf / np.linalg.norm(psf)

    # Remove defective pixels in mask calibration
    mask_sum = np.sum(mask, 2)
    ind = np.unravel_index((np.argmax(mask_sum, axis=None)), mask_sum.shape)
    mask[ind[0] - 2 : ind[0] + 2, ind[1] - 2 : ind[1] + 2, :] = 0

    # Remove defective pixels in measurement
    im = im[c1:c2, c3:c4]
    im = im / np.max(im)
    im[ind[0] - 2 : ind[0] + 2, ind[1] - 2 : ind[1] + 2] = 0
    return mask, psf, im


def interp_along_axis(y, x, newx, axis, inverse=False, method="linear"):
    """Interpolate vertical profiles, e.g. of atmospheric variables
    using vectorized numpy operations

    This function assumes that the x-xoordinate increases monotonically

    ps:
    * Updated to work with irregularly spaced x-coordinate.
    * Updated to work with irregularly spaced newx-coordinate
    * Updated to easily inverse the direction of the x-coordinate
    * Updated to fill with nans outside extrapolation range
    * Updated to include a linear interpolation method as well
        (it was initially written for a cubic function)

    Peter Kalverla
    March 2018

    --------------------
    More info:
    Algorithm from: http://www.paulinternet.nl/?page=bicubic
    It approximates y = f(x) = ax^3 + bx^2 + cx + d
    where y may be an ndarray input vector
    Returns f(newx)

    The algorithm uses the derivative f'(x) = 3ax^2 + 2bx + c
    and uses the fact that:
    f(0) = d
    f(1) = a + b + c + d
    f'(0) = c
    f'(1) = 3a + 2b + c

    Rewriting this yields expressions for a, b, c, d:
    a = 2f(0) - 2f(1) + f'(0) + f'(1)
    b = -3f(0) + 3f(1) - 2f'(0) - f'(1)
    c = f'(0)
    d = f(0)

    These can be evaluated at two neighbouring points in x and
    as such constitute the piecewise cubic interpolator.
    """

    # View of x and y with axis as first dimension
    if inverse:
        _x = np.moveaxis(x, axis, 0)[::-1, ...]
        _y = np.moveaxis(y, axis, 0)[::-1, ...]
        _newx = np.moveaxis(newx, axis, 0)[::-1, ...]
    else:
        _y = np.moveaxis(y, axis, 0)
        _x = np.moveaxis(x, axis, 0)
        _newx = np.moveaxis(newx, axis, 0)

    # Sanity checks
    if np.any(_newx[0] < _x[0]) or np.any(_newx[-1] > _x[-1]):
        # raise ValueError('This function cannot extrapolate')
        warnings.warn(
            "Some values are outside the interpolation range. "
            "These will be filled with NaN"
        )
    if np.any(np.diff(_x, axis=0) < 0):
        raise ValueError("x should increase monotonically")
    if np.any(np.diff(_newx, axis=0) < 0):
        raise ValueError("newx should increase monotonically")

    # Cubic interpolation needs the gradient of y in addition to its values
    if method == "cubic":
        # For now, simply use a numpy function to get the derivatives
        # This produces the largest memory overhead of the function and
        # could alternatively be done in passing.
        ydx = np.gradient(_y, axis=0, edge_order=2)

    # This will later be concatenated with a dynamic '0th' index
    ind = [i for i in np.indices(_y.shape[1:])]

    # Allocate the output array
    original_dims = _y.shape
    newdims = list(original_dims)
    newdims[0] = len(_newx)
    newy = np.zeros(newdims)

    # set initial bounds
    i_lower = np.zeros(_x.shape[1:], dtype=int)
    i_upper = np.ones(_x.shape[1:], dtype=int)
    x_lower = _x[0, ...]
    x_upper = _x[1, ...]

    for i, xi in enumerate(_newx):
        # Start at the 'bottom' of the array and work upwards
        # This only works if x and newx increase monotonically

        # Update bounds where necessary and possible
        needs_update = (xi > x_upper) & (i_upper + 1 < len(_x))
        # print x_upper.max(), np.any(needs_update)
        while np.any(needs_update):
            i_lower = np.where(needs_update, i_lower + 1, i_lower)
            i_upper = i_lower + 1
            x_lower = _x[[i_lower] + ind]
            x_upper = _x[[i_upper] + ind]

            # Check again
            needs_update = (xi > x_upper) & (i_upper + 1 < len(_x))

        # Express the position of xi relative to its neighbours
        xj = (xi - x_lower) / (x_upper - x_lower)

        # Determine where there is a valid interpolation range
        within_bounds = (_x[0, ...] < xi) & (xi < _x[-1, ...])

        if method == "linear":
            f0, f1 = _y[[i_lower] + ind], _y[[i_upper] + ind]
            a = f1 - f0
            b = f0

            newy[i, ...] = np.where(within_bounds, a * xj + b, np.nan)

        elif method == "cubic":
            f0, f1 = _y[[i_lower] + ind], _y[[i_upper] + ind]
            df0, df1 = ydx[[i_lower] + ind], ydx[[i_upper] + ind]

            a = 2 * f0 - 2 * f1 + df0 + df1
            b = -3 * f0 + 3 * f1 - 2 * df0 - df1
            c = df0
            d = f0

            newy[i, ...] = np.where(
                within_bounds, a * xj**3 + b * xj**2 + c * xj + d, np.nan
            )

        else:
            raise ValueError(
                "invalid interpolation method" "(choose 'linear' or 'cubic')"
            )

    if inverse:
        newy = newy[::-1, ...]

    return np.moveaxis(newy, 0, axis)
