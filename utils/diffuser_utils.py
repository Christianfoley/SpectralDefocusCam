import torch.nn.functional as F
import os, tqdm
import numpy as np
import torch
import torch.optim
import scipy.io
import cv2

import models.rdmpy.blur as blur


def interleave3d(t, spec_dim=2):  # takes 1,2,2,x,a,a returns 1,2,2x,a,a
    assert spec_dim > 0  # we need 0th dimension to be the blurs
    spec_dim = spec_dim - 1
    batchsize = len(t)
    outlist = []
    for i in range(batchsize):
        ab = t[i]
        stacked = torch.stack([ab[0], ab[1]], dim=spec_dim)
        interleaved = torch.flatten(stacked, start_dim=spec_dim - 1, end_dim=spec_dim)
        outlist.append(interleaved)
    return torch.stack(outlist)


def fft_psf(model, h):
    h_complex = pad_zeros_torch(model, torch.complex(h, torch.zeros_like(h)))
    H = torch.fft.fft2(torch.fft.ifftshift(h_complex)).unsqueeze(1)
    return H


def fft_im(im):
    xc = torch.complex(im, torch.zeros_like(im))
    Xi = torch.fft.fft2(xc)
    return Xi


def tt(x, device="cuda:0"):
    return torch.tensor(x, dtype=torch.float32, device=device)


def pyramid_down(image, out_shape):
    if image.shape[0] == out_shape[0] and image.shape[1] == out_shape[1]:
        return image
    closest_pyr = cv2.pyrDown(image, (image.shape[0] // 2, image.shape[1] // 2))
    return cv2.resize(closest_pyr, out_shape, interpolation=cv2.INTER_AREA)


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
    Crutch function to perform 2d ring convolution on every 2d slice in a
    batched hyperspectral blur stack.

    Note that if the psf batch is too large in r and h, the batch's x and y dimensions
    will be upsampled to match

    Parameters
    ----------
    batch : torch.Tensor
        5d tensor of shape (batch, n_blur, channel, y, x)
    psfs : torch.Tensor
        4d tensor of shape (n_blur, r, theta, h)
    device : torch.device, optional
        device, by default cpu

    Returns
    -------
    torch.Tensor
        blurred batch
    """
    assert len(batch.shape) == 5, "batch must be of shape b, n, c, y, x"
    assert len(psfs.shape) == 4, "psf data must be of shape n, r, theta, h"

    # fix size mismatch issue (we upsample to psfs->conv->downsample back)
    batch = batch.type(torch.float32)
    batch, output_size = rescale_to_psfs(batch, (psfs.shape[1], psfs.shape[3]))

    convolved_batch = torch.zeros_like(batch, device=device)
    for n in range(batch.shape[1]):
        convolved_batch[:, n, ...] = blur.batch_ring_convolve(
            batch[:, n, ...], psfs[n], device=device
        )

    batch, _ = rescale_to_psfs(convolved_batch, output_size)
    return batch


def load_mask(
    path="/home/cfoley_waller/defocam/defocuscamdata/calibration_data/calibration.mat",
    patch_crop_center=[500, 1500],  # [200, 2100],
    patch_crop_size=[768, 768],
    patch_size=[256, 256],
    old_calibration=False,
    sum_chans_2=True,
):
    spectral_mask = scipy.io.loadmat(path)
    dims = (
        patch_crop_center[0] - patch_crop_size[0] // 2,
        patch_crop_center[0] + patch_crop_size[0] // 2,
        patch_crop_center[1] - patch_crop_size[1] // 2,
        patch_crop_center[1] + patch_crop_size[1] // 2,
    )
    mask = spectral_mask["mask"][dims[0] : dims[1], dims[2] : dims[3]]

    if old_calibration:  # weird things about models trained with calibration matrix
        mask = mask[100 : 100 + patch_size[0], 100 : 100 + patch_size[1], :-1]
        mask = (mask[..., 0::2] + mask[..., 1::2]) / 2
        mask = mask[..., 0:30]
    else:
        # widening "filter" resolution to adjascent wavelengths
        if sum_chans_2:
            mask1 = mask[..., 0:-1:2]
            mask2 = mask[..., 1::2]
            mask = (mask1 + mask2)[..., 0:30]
        else:
            mask = mask[..., :64]

        # downsample & normalize
        nm = lambda x: (x - np.min(x)) / (np.max(x - np.min(x)))
        mask = np.stack(
            [pyramid_down(mask[..., i], patch_size) for i in range(mask.shape[-1])],
            axis=-1,
        )

        mask = nm(mask)
    return mask


def process_mat_files(input_directory, output_directory, overwrite=True):
    """
    Process .mat files from the input directory, extracting the 'image' array and saving it as a new .mat file in the output directory.

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
