import itertools
import numpy as np
import os, glob, csv
import scipy.io as io
import h5py
from scipy.interpolate import interp1d
import tqdm
import ray


def get_patches(patch_size, img_size):
    """
    Given a patchsize and an image size, will return a list of patches (top to bottom,
    left to right) of size patch_size. If patch exceeds image boundary, will ignore

    Parameters
    ----------
    patch_size : tuple
        y and x dimensions of size of each patch
    img_size : tuple
        y and x dimensions of image to be patched

    Returns
    -------
    list
        list of patch tuples (y0, y1, x0, x1)
    """
    patch_size, img_size = patch_size[0:2], img_size[0:2]
    patches = []
    for i in range(0, img_size[0], patch_size[0]):
        for j in range(0, img_size[1], patch_size[1]):
            if i + patch_size[0] > img_size[0] or j + patch_size[1] > img_size[1]:
                continue
            patches.append((i, i + patch_size[0], j, j + patch_size[1]))
    return patches


def save_patches(patches, outpath, sourcepath, overwrite_existing) -> list[str]:
    """
    Save a list of patches to a directory outpath that were created from a file at sourcepath.
    Files are all saved in the outpath as derivatives of their sourcepath. Specifically, they are
    saved as "sourcepath_name + "_" + "patch_number".

    Parameters
    ----------
    patches : list
        list of patch data (np.ndarray)
    outpath : str
        path to output dir
    sourcepath : str
        path to img data file from which patches were created
    overwrite_existing : bool
        whether to overwrite existing data in outpath
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    filename = os.path.splitext(os.path.basename(sourcepath))[0]

    output_patch_paths = []
    for i, patch in enumerate(patches):
        patch_dict = {"image": patch}
        output_filename = f"{filename}_patch_{i}.mat"
        output_path = os.path.join(outpath, output_filename)

        if os.path.exists(output_path) and not overwrite_existing:
            print(f"Patch {output_filename} already exists, skipping...")
            output_patch_paths.append(output_path)
            continue

        io.savemat(output_path, patch_dict)
        output_patch_paths.append(output_path)
    return output_patch_paths


def read_compressed(img):
    """
    Decompress a PCA-compressed image sample (for use in hyperspectral fruit data)

    Parameters
    ----------
    img : dict
        image sample as a dict. contains keys "wc", "pcc", "wid", "hei"

    Returns
    -------
    np.ndarray
        decompressed image sample
    """
    wc, pcc, wid, hei = img["wc"], img["pcc"], img["wid"], img["hei"]
    spectra = np.matmul(pcc, np.transpose(wc))

    decompressed = np.reshape(
        np.transpose(spectra)[:, :, None], (wid[0][0], hei[0][0], len(spectra))
    )
    return decompressed


def project_spectral(img, out_channels, chan_range=None):
    """
    Project the image from its base spectral space to one with "out_channels" bands via linear
    interpolation from specified a range of spectral bands. By default, uses all bands

    Parameters
    ----------
    img : np.ndarray
        hyperspectral image (y, x, lambda)
    out_channels : int
        dimension of output spectral space
    chan_range : tuple, optional
        range of spectral channels in original image to consider, by default None
    """
    if chan_range is not None:
        img = img[:, :, chan_range[0] : chan_range[1]]

    x_vals = np.linspace(0, 1, img.shape[2])
    projection_x_vals = np.linspace(0, 1, out_channels)

    f_out = interp1d(x_vals, img, axis=2)
    return f_out(projection_x_vals)


def preprocess_harvard_img(
    img, patch_size, calib_vec, num_channels, apply_calib=True, skip_masked=True
):
    """
    Preprocesses harvard images into patches of given size and returns a list of patch
    data. When patching will ignore patches which contain part of the mask.

    Parameters
    ----------
    img : dict
        dict containing 'ref' and 'lbl' keys for image data and mask
    patch_size : tuple
        desired size of patches
    calib_vec : list
        list of sensitivity floats for each channel in img
    num_channels : int
        number of spectral channels desired in output image
    """
    img_data, mask_data = img["ref"], img["lbl"]
    calib_vec = np.expand_dims(np.array(calib_vec), axis=(0, 1))
    if not apply_calib:
        calib_vec = np.ones_like(calib_vec)
    patches = get_patches(patch_size, img_data.shape)

    img_patches = []
    for p in patches:
        mask_patch = mask_data[p[0] : p[1], p[2] : p[3]]
        img_patch = img_data[p[0] : p[1], p[2] : p[3]]
        if np.min(mask_patch) == 0 and skip_masked:
            continue

        img_patch = (img_patch) / calib_vec
        img_patch = project_spectral(img_patch, num_channels)

        img_patch = img_patch.astype(np.float32)
        img_patch = img_patch / np.max(img_patch)
        img_patches.append(img_patch)

    return img_patches


@ray.remote(num_cpus=1)
def _preprocess_harvard_img_ray(
    img_path: str,
    out_path: str,
    patch_size: tuple[int, int],
    calib_vec: list,
    num_channels: int,
    skip_masked: bool = True,
    overwrite_existing: bool = False,
) -> list[str]:
    """
    Helper to intercept and distribute processing of Harvard images.
    """
    sourcepath = img_path
    try:
        img = io.loadmat(img_path)
        patches = preprocess_harvard_img(
            img, patch_size, calib_vec, num_channels, skip_masked=skip_masked
        )
        return save_patches(patches, out_path, sourcepath, overwrite_existing)
    except Exception as e:
        print(f"Skipping {os.path.basename(img_path)}:{e}")
    return []


def preprocess_harvard_data(
    datapath,
    outpath,
    patch_size,
    num_channels=30,
    skip_masked=True,
    overwrite_existing=False,
):
    """
    Preprocesses all harvard data into patches. Interpolates along spectral dimension and
    saves each patch as a .mat file.
    For specifics see http://vision.seas.harvard.edu/hyperspec/

    Parameters
    ----------
    datapath : str
        path to folder containing CZ_hsdb and CZ_hsdbi folders
    outpath : str
        destination data folder
    patch_size : tuple
        x and y patch size
    num_channels : int
        number of spectral channels
    """
    assert ray.is_initialized(), "Ray must be initialized to preprocess Harvard data."
    os.makedirs(outpath, exist_ok=True)

    imgs_hsdbi = glob.glob(os.path.join(datapath, "Cz_hsdbi/*.mat"))
    with open(os.path.join(datapath, "CZ_hsdbi/calib.txt"), "r") as f:
        calib_vals_hsdbi = [float(num) for num in f.readline().split("   ")[1:]]

    imgs_hsdb = glob.glob(os.path.join(datapath, "CZ_hsdb/*.mat"))
    with open(os.path.join(datapath, "CZ_hsdb/calib.txt"), "r") as f:
        calib_vals_hsdb = [float(num) for num in f.readline().split("   ")[1:]]

    all_result_paths = []
    for i, img in tqdm.tqdm(
        list(enumerate(imgs_hsdb + imgs_hsdbi)), desc="Preprocessing Harvard Data"
    ):
        calib_vals = calib_vals_hsdb if i < len(imgs_hsdb) else calib_vals_hsdbi
        result_paths_promise = _preprocess_harvard_img_ray.remote(
            img,
            outpath,
            patch_size,
            calib_vals,
            num_channels,
            skip_masked,
            overwrite_existing,
        )
        all_result_paths.append(result_paths_promise)
    all_result_paths = ray.get(all_result_paths)
    return list(itertools.chain.from_iterable(all_result_paths))


def preprocess_pavia_img(img, patch_size, num_channels):
    """
    Preprocesses pavia images into patches of given size and returns a list of patch
    data.

    Parameters
    ----------
    img : dict
        dict containing 'ref' and 'lbl' keys for image data and mask
    patch_size : tuple
        desired size of patches
    num_channels : int
        number of spectral channels in output
    """
    if "pavia" in img:
        img_data = img["pavia"]
    else:
        img_data = img["paviaU"]
    patches = get_patches(patch_size, img_data.shape)

    img_patches = []
    for p in patches:
        img_patch = img_data[p[0] : p[1], p[2] : p[3]]
        img_patch = project_spectral(img_patch, num_channels)

        img_patch = img_patch.astype(np.float32)
        img_patch = img_patch / np.max(img_patch)
        img_patches.append(img_patch)

    return img_patches


@ray.remote(num_cpus=1)
def _preprocess_pavia_img_ray(
    img_path: str,
    out_path: str,
    patch_size: tuple[int, int],
    num_channels: int,
    overwrite_existing: bool = False,
) -> list[str]:
    """
    Helper to intercept and distribute processing of Pavia images.
    """
    sourcepath = img_path
    try:
        img = io.loadmat(img_path)
        patches = preprocess_pavia_img(img, patch_size, num_channels)
        return save_patches(patches, out_path, sourcepath, overwrite_existing)
    except Exception as e:
        print(f"Skipping {os.path.basename(img_path)}:{e}")
    return []


def preprocess_pavia_data(
    datapath, outpath, patch_size, num_channels=30, overwrite_existing=False
):
    """
    Preprocesses all pavia data into patches. Interpolates along spectral dimension and
    saves each patch as a .mat file.
    For specifics see:
    https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

    Parameters
    ----------
    datapath : str
        path to folder containing pavia ".mat" files
    outpath : str
        destination data folder
    patch_size : tuple
        x and y patch size
    num_channels : int
        number of spectral channels in output
    """
    assert ray.is_initialized(), "Ray must be initialized to preprocess Pavia data."
    os.makedirs(outpath, exist_ok=True)
    imgs = glob.glob(os.path.join(datapath, "*.mat"))

    all_result_paths = []
    for img in tqdm.tqdm(imgs, desc="Preprocessing Pavia Data"):
        result_paths_promise = _preprocess_pavia_img_ray.remote(
            img, outpath, patch_size, num_channels, overwrite_existing
        )
        all_result_paths.append(result_paths_promise)
    all_result_paths = ray.get(all_result_paths)
    return list(itertools.chain.from_iterable(all_result_paths))


def preprocess_fruit_img(img, patch_size, num_channels):
    """
    Preprocesses hyperspectral fruit images into patches of given size and
    returns a list of patch data.

    Parameters
    ----------
    img : dict
        dict containing "wc", "pcc", "wid", "hei" keys for pca compressed data
    patch_size : tuple
        desired size of patches
    num_channels : int
        number of spectral channels in output
    """
    img_data = read_compressed(img)
    patches = get_patches(patch_size, img_data.shape)

    img_patches = []
    for p in patches:
        img_patch = img_data[p[0] : p[1], p[2] : p[3]]
        img_patch = project_spectral(img_patch, num_channels)

        img_patch = img_patch.astype(np.float32)
        img_patch = img_patch / np.max(img_patch)
        img_patches.append(img_patch)

    return img_patches


@ray.remote(num_cpus=1)
def _preprocess_fruit_img_ray(
    img_path: str,
    out_path: str,
    patch_size: tuple[int, int],
    num_channels: int,
    overwrite_existing: bool = False,
) -> list[str]:
    """
    Helper to intercept and distribute processing of fruit images. See
    `preprocess_fruit_img` docstring.
    """
    sourcepath = img_path
    try:
        img = io.loadmat(img_path)
        patches = preprocess_fruit_img(img, patch_size, num_channels)
        return save_patches(patches, out_path, sourcepath, overwrite_existing)
    except Exception as e:
        print(f"Skipping {os.path.basename(img_path)}:{e}")
    return []


def preprocess_fruit_data(
    datapath, outpath, patch_size, num_channels=30, overwrite_existing=False
):
    """
    Preprocesses all HS fruit data into patches. Interpolates along spectral dimension and
    saves each patch as a .mat file.
    For specifics see: https://zenodo.org/record/2611806

    Parameters
    ----------
    datapath : str
        path to folder containing compressed fruit ".mat" colorspace files
    outpath : str
        destination data folder
    patch_size : tuple
        x and y patch size
    num_channels : int
        number of spectral channels in output
    overwrite_existing : bool
        whether to overwrite existing data in outpath
    """
    assert ray.is_initialized(), "Ray must be initialized to preprocess fruit data."
    os.makedirs(outpath, exist_ok=True)

    imgs = glob.glob(os.path.join(datapath, "*.mat"))

    all_result_paths = []
    for img in tqdm.tqdm(imgs, desc="Preprocessing Fruit Data"):
        result_paths_promise = _preprocess_fruit_img_ray.remote(
            img, outpath, patch_size, num_channels, overwrite_existing
        )
        all_result_paths.append(result_paths_promise)
    all_result_paths = ray.get(all_result_paths)

    return list(itertools.chain.from_iterable(all_result_paths))


def preprocess_icvl_img(img, patch_size, num_channels):
    """
    Preprocesses hyperspectral scenes into patches of given size and
    returns a list of patch data.

    Parameters
    ----------
    img : dict or hd5py container
        dict containing "rad" key for image data
    patch_size : tuple
        desired size of patches
    num_channels : int
        number of spectral channels in output
    """
    img_data = np.array(img["rad"]).transpose(1, 2, 0)[::-1, ::-1]
    patches = get_patches(patch_size, img_data.shape)

    img_patches = []
    for p in patches:
        img_patch = img_data[p[0] : p[1], p[2] : p[3]]
        # NOTE fudge factor for speed reasons: 31 -> 30 channels
        # img_patch = project_spectral(img_patch, num_channels)
        img_patch = img_patch[:, :, :-1]

        img_patch = img_patch.astype(np.float32)
        img_patch = img_patch / np.max(img_patch)
        img_patches.append(img_patch)

    return img_patches


@ray.remote(num_cpus=1)
def _preprocess_icvl_img_ray(
    img_path: str,
    out_path: str,
    patch_size: tuple[int, int],
    num_channels: int,
    overwrite_existing: bool = False,
) -> list[str]:
    """
    Helper to intercept and distribute processing of ICVL images. See
    `preprocess_icvl_img` docstring.
    """
    sourcepath = img_path
    try:
        img = h5py.File(img_path)

        patches = preprocess_icvl_img(img, patch_size, num_channels)
        return save_patches(patches, out_path, sourcepath, overwrite_existing)
    except Exception as e:
        print(f"Skipping {os.path.basename(img_path)}:{e}")

    return []


def preprocess_icvl_data(
    datapath, outpath, patch_size, num_channels=30, overwrite_existing=False
):
    """
    Preprocess ICVL image into simulation-ready patches of given size.
    For specifics see: https://icvl.cs.bgu.ac.il/hyperspectral/

    Parameters
    ----------
    datapath : str
        path to folder containing compressed fruit ".mat" colorspace files
    outpath : str
        destination data folder
    patch_size : tuple
        x and y patch size
    num_channels : int
        number of spectral channels in output
    overwrite_existing : bool
        whether to overwrite existing data in outpath
    """
    assert ray.is_initialized(), "Ray must be initialized to preprocess ICVL data."
    os.makedirs(outpath, exist_ok=True)

    imgs = glob.glob(os.path.join(datapath, "*.mat"))

    all_result_paths = []
    for img in tqdm.tqdm(imgs, desc="Preprocessing ICVL Data"):
        result_paths_promise = _preprocess_icvl_img_ray.remote(
            img, outpath, patch_size, num_channels, overwrite_existing
        )
        all_result_paths.append(result_paths_promise)
    all_result_paths = ray.get(all_result_paths)

    return list(itertools.chain.from_iterable(all_result_paths))
