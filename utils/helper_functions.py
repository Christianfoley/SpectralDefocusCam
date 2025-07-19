import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io
import yaml
import colour
from IPython.core.display import display, HTML
from ipywidgets import interact, widgets, fixed

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

import sys

EPSILON = 2e-12

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
    lib = np
    if isinstance(x, torch.Tensor):
        lib = torch
    x = x - lib.min(x)
    return x / (lib.max(x) + EPSILON)


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


# ----------------- VARIOUS METHODS FOR FALSE COLOR GENERATION ----------------- #


def _resample_spectral_cube(cube, wavelengths, step=10):
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

    flat_cube = cube.reshape(-1, L)
    new_flat_cube = np.array(
        [np.interp(new_wavelengths, wavelengths, spectrum) for spectrum in flat_cube]
    )

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
    cube, new_wavs = _resample_spectral_cube(data_cube, wavs, step)

    # Ensure data is in reasonable range (normalize if needed)
    # Assuming your spectral data represents reflectance (0-1) or radiance
    if np.max(cube) > 10:
        cube = cube / np.max(cube)

    pixels = cube.reshape(-1, cube.shape[-1])  # shape: (H*W, L)
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
    illuminated_pixels = pixels * illum_array[np.newaxis, :]  # Broadcast illuminant

    # Integrate to get XYZ values
    # Using trapezoidal integration approximation
    XYZ = k * wavelength_interval * np.dot(illuminated_pixels, cmf_array)
    rgb_linear = colour.XYZ_to_sRGB(XYZ / 100.0)  # This gives linear RGB values

    # Handle out-of-gamut colors more gracefully with custom gamma correction
    rgb_linear = np.clip(rgb_linear, 0, 1)
    gamma_corrected = np.power(rgb_linear, 1.0 / gamma)
    rgb_final = np.clip(gamma_corrected, 0, 1)

    return rgb_final.reshape(H, W, 3)


def select_and_average_bands(
    data_cube,
    fc_range=(450, 810),
    spectral_ranges=[(400, 495), (495, 600), (600, 750)],
    scaling=[1, 1, 1],
):
    """
    Naive method that just averages over rgb bands in the data cube
    and returns an RGB image.
    """
    wavs = np.linspace(fc_range[0], fc_range[1], data_cube.shape[-1])

    averaged_bands = []
    for spectral_range in spectral_ranges:
        indices = np.where((wavs >= spectral_range[0]) & (wavs <= spectral_range[1]))[0]

        if len(indices) > 0:
            averaged_band = np.mean(data_cube[:, :, indices], axis=2)
        else:
            averaged_band = np.zeros_like(data_cube[:, :, 0])
        averaged_bands.append(averaged_band)
    averaged_bands = [averaged_bands[i] * scaling[i] for i in range(3)]

    # Stack the averaged bands along the last dimension to form an RGB image
    rgb_image = np.stack(averaged_bands, axis=-1)[:, :, ::-1]
    return rgb_image


def stack_rgb_opt_30(
    reflArray,
    fc_range=[450, 810],
    offset=0,
    scaling=[1, 1, 1],
):
    """
    Projects a hyperspectral array onto false color and returns a 3d
    rgb image. hyperspectral image assumed to have a start and end wavelength
    starting at index "offset" specified by "range".

    Parameters
    ----------
    reflArray : np.ndarray
        hyperspectral image (y,x,lambda)
    fc_range : list, optional
        range from lowest to highest wavelength, by default [450, 810]
    offset : int, optional
        offset index in reflarray for start of range, by default 0
    scaling : list, optional
        scaling of each color curve, by default [1, 1, 2.5]

    Returns
    -------
    np.ndarray
        false color RGB image, (y,x,3)
    """
    # Validate input
    if len(fc_range) != 2:
        raise ValueError(
            "spectral_range should have two elements (start and end wavelengths)."
        )
    if len(scaling) != 3:
        raise ValueError(
            "scaling should have three elements (one for each RGB channel)."
        )
    if reflArray.ndim != 3:
        raise ValueError("reflArray should be a 3D numpy array.")

    # load color matching functions
    cmfs = colour.colorimetry.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

    # enforce offset and fix bounds
    reflArray = reflArray[:, :, offset:]
    fc_range = [
        max(fc_range[0], cmfs.wavelengths[0]),
        min(fc_range[1], cmfs.wavelengths[-1]),
    ]

    # align the cmfs to the requested range and integrate over channels
    wavelengths = np.linspace(*fc_range[:2], reflArray.shape[2], endpoint=False)
    idcs = np.squeeze(
        np.array([np.where(cmfs.wavelengths == int(w)) for w in wavelengths])
    )

    rgb_image = np.einsum("yxwc,wc->yxc", reflArray[..., None], cmfs.values[idcs, :])

    # scale
    rgb_image = rgb_image * np.array(scaling)[None, None, :] / len(wavelengths)
    return rgb_image


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


def plot_superpixel_waves(
    cube, waves_start=390, waves_end=870, startx=1557, starty=826, scale=1
):
    """
    Plots wavelengths from every filter in a single superpixel

    Helper tool for analyzing filter calibrations.
    """
    sp_size, filt_size, offset = 66 / scale, 8.3 / scale, 4 / scale

    superpix = cube[:, starty : int(starty + sp_size), startx : int(startx + sp_size)]

    temp_img = np.mean(superpix, 0)
    wavs = np.linspace(waves_start, waves_end, cube.shape[0])

    # plot waves and their origins
    fig, ax = plt.subplots(1, 2, figsize=(17, 8))
    maxval = np.max(temp_img)
    for i in range(0, 8):
        for j in range(0, 8):
            ax[0].plot(
                wavs,
                np.mean(
                    superpix[
                        :,
                        int(offset + i * filt_size)
                        - 1 : int(offset + i * filt_size)
                        + 2,
                        int(offset + j * filt_size)
                        - 1 : int(offset + j * filt_size)
                        + 2,
                    ],
                    (-1, -2),
                ),
            )
            temp_img[int(offset + i * filt_size), int(offset + j * filt_size)] = (
                maxval * 1.2
            )

    ax[1].imshow(temp_img)
    ax[0].set_title("Filter waves (nm)")
    ax[1].set_title("Sample origins")
    plt.show()


def plot_cube_interactive(
    data_cube,
    height=600,
    width=1200,
    use_false_color=True,
    fc_range=[450, 810],
    fc_scaling=[1, 1, 1],
    avg_block_size=1,
):
    """
    Returns an interactive plotly figure using ipywidgets
    that plots a hyperspectral image's response at the clicked pixel

    Parameters
    ----------
    data_cube : np.ndarray
        3d hyperspectral data cube (y,x,lambda)
    height : int, optional
        height of plot, by default 600
    width : int, optional
        width of plot, by default 1200
    use_false_color : bool, optional
        whether to use FC images
    fc_range : list, optional
        range of wavelengths in data cube (start, end), by default [450, 810]
    fc_scaling : list, optional
        scaling of FC channels (r g b), by default [1, 1, 1]
    avg_block_size : int, optional
        size of block around clicked pixel to average
    Returns
    -------
    go.FigureWidget
        interactive plotly widget figure
    """
    mean_image = np.mean(data_cube, axis=2)

    # init plot with the fc image, an empty vector plot, and a marker trace
    wavs = np.linspace(*fc_range, data_cube.shape[-1])
    fig = go.FigureWidget(
        make_subplots(rows=1, cols=2, subplot_titles=["False Color Image", "Response"])
    )
    if use_false_color:
        projected_false_color = (
            value_norm(
                select_and_average_bands(data_cube, fc_range, scaling=fc_scaling)
            )
            * 255
        ).astype(np.uint8)
        image_trace = go.Image(z=projected_false_color)
    else:
        image_trace = go.Heatmap(z=mean_image, colorscale="Viridis")
    vector_plot_trace = go.Scatter(
        x=wavs, y=[], mode="lines+markers", name="point response"
    )
    vector_plot_trace_mean = go.Scatter(
        x=wavs,
        y=data_cube.mean(axis=(0, 1)),
        mode="lines+markers",
        name=f"local {avg_block_size} pix response",
    )
    marker_trace = go.Scatter(
        x=[], y=[], mode="markers", marker=dict(color="red", size=10)
    )

    fig.add_trace(image_trace, row=1, col=1)
    fig.add_trace(vector_plot_trace, row=1, col=2)
    fig.add_trace(vector_plot_trace_mean, row=1, col=2)
    fig.add_trace(marker_trace, row=1, col=1)
    fig.update_layout(
        height=height,
        width=width,
        title_text="Click on image to view response vector",
    )
    fig.update_yaxes(range=[0, np.max(data_cube)], row=1, col=2)
    fig.update_yaxes(range=[0, np.max(data_cube)], row=1, col=2)

    # Function to update the plot based on click
    def update_plot_on_click(trace, points, selector):
        """On-click update function for plot"""
        if points.xs and points.ys:
            x, y = int(points.xs[0]), int(points.ys[0])
            bs = avg_block_size // 2

            # Define the slice ranges
            y_start = max(y - bs, 0)
            y_end = min(y + bs + 1, data_cube.shape[0])
            x_start = max(x - bs, 0)
            x_end = min(x + bs + 1, data_cube.shape[1])

            # Extract the region from data_cube and compute mean
            depth_vector = data_cube[y, x, :]
            depth_vector_mean = data_cube[y_start:y_end, x_start:x_end, :].mean(
                axis=(0, 1)
            )

            # Update plot and img marker
            with fig.batch_update():
                fig.data[1].y = depth_vector
                fig.data[2].y = depth_vector_mean
                fig.layout.annotations[1].text = f"Response at ({x}, {y})"

                fig.data[3].x = [x]
                fig.data[3].y = [y]

    # Attach the click event to the heatmap
    fig.data[0].on_click(update_plot_on_click)
    return fig


def plot_cube_3d_scatter(
    data_cube,
    maxval_thresh=0.5,
    quantile_thresh=0.7,
    point_opacity=0.9,
    point_size=2,
    downsample_yx_scale=2,
    cscale="Jet",
):
    """
    Generates interactive 3D scatter plot of hyperspectral cube.
    Thresholds out all values for each x/y position depending on
    quantile or maxval

    Parameters
    ----------
    data_cube : np.ndarray
        3d data cube to visualize (y,x,lambda)
    maxval_thresh : float, optional
        proportion of max response of each x/y to thresh, by default 0.5
    quantile_thresh : int, optional
        quantile of max response of each x/y to filter, by default 0
    point_opacity : float, optional
        opacity of points, by default 0.9
    point_size : int, optional
        size of points, by default 2
    downsample_yx_scale : int, optional
        scale of downsampling to use in y and x, by default 2
    cscale : str, optional
        color mapping, by default "Jet"
    """
    # Initialize Plotly for Jupyter Notebook
    init_notebook_mode(connected=True)
    if downsample_yx_scale:
        data_cube = data_cube[::downsample_yx_scale, ::downsample_yx_scale]

    threshold = np.maximum(
        np.quantile(data_cube, quantile_thresh),
        np.max(data_cube, axis=2, keepdims=True) * maxval_thresh,
    )
    x, y, z = np.where(data_cube > threshold)
    values = data_cube[x, y, z]

    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=point_size,
            color=values,
            colorscale=cscale,
            opacity=point_opacity,
        ),
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X-axis"),
            yaxis=dict(title="Y-axis"),
            zaxis=dict(title="Lambda-axis"),
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Adjust margins for a cleaner plot
    )

    fig = go.Figure(data=[trace], layout=layout)
    return iplot(fig)
