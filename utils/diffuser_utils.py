import torch.nn.functional as F
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


def batch_ring_convolve(data, psfs, device=torch.device("cpu")):
    """
    Crutch function to perform 2d ring convolution on every 2d slice in a
    batched hyperspectral blur stack

    Parameters
    ----------
    data : torch.Tensor
        5d tensor of shape (batch, n_blur, channel, y, x)
    psfs : torch.Tensor
        4d tensor of shape (n_blur, y, psfs, x)
    device : torch.device, optional
        device, by default cpu

    Returns
    -------
    torch.Tensor
        blurred batch
    """
    # TODO find a better way to do this (maybe reimplementing blur.py)
    assert len(data.shape) == 5, "data must be of shape b, n, c, y, x"
    assert len(psfs.shape) == 4, "psf data must be of shape n, c, y, x"

    batch = []
    for b in range(data.shape[0]):
        blur_stack = []
        for n in range(data.shape[1]):
            channel = []
            for c in range(data.shape[2]):
                blurred = blur.ring_convolve(data[b, n, c], psfs[n], device=device)
                channel.append(blurred)
            blur_stack.append(torch.stack(channel, 0))
        batch.append(torch.stack(blur_stack, 0))
    return batch


def load_mask(
    path="/home/cfoley_waller/defocam/defocuscamdata/calibration_data/calibration.mat",
    patch_crop_source=[500, 1500],  # [200, 2100],
    patch_crop_size=[768, 768],
    patch_size=[256, 256],
    old_calibration=False,
    sum_chans_2=True,
):
    spectral_mask = scipy.io.loadmat(path)
    dims = (
        patch_crop_source[0],
        patch_crop_source[0] + patch_crop_size[0],
        patch_crop_source[1],
        patch_crop_source[1] + patch_crop_size[1],
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


# def crop_forward2(model, x):
#    C01 = model.PAD_SIZE0//2; C02 = model.PAD_SIZE0//2 + model.DIMS0//2              # Crop indices
#    C11 = model.PAD_SIZE1//2; C12 = model.PAD_SIZE1//2 + model.DIMS1//2              # Crop indices
#    return x[:, :, :, C01:C02, C11:C12]
class Forward_Model(torch.nn.Module):
    def __init__(self, h_in, shutter=0, cuda_device=0):
        super(Forward_Model, self).__init__()
        self.cuda_device = cuda_device

        ## Initialize constants
        self.DIMS0 = h_in.shape[0]  # Image Dimensions
        self.DIMS1 = h_in.shape[1]  # Image Dimensions
        self.PAD_SIZE0 = int((self.DIMS0))  # Pad size
        self.PAD_SIZE1 = int((self.DIMS1))  # Pad size

        #         self.h_var = torch.nn.Parameter(torch.tensor(h_in, dtype=torch.float32, device=self.cuda_device),
        #                                             requires_grad=False)
        #         self.h_zeros = torch.nn.Parameter(torch.zeros(self.DIMS0*2, self.DIMS1*2, dtype=torch.float32, device=self.cuda_device),
        #                                           requires_grad=False)
        self.h_complex = pad_zeros_torch(
            self,
            torch.tensor(h_in, dtype=torch.cfloat, device=self.cuda_device).unsqueeze(
                0
            ),
        )
        self.const = torch.tensor(
            1 / np.sqrt(self.DIMS0 * 2 * self.DIMS1 * 2),
            dtype=torch.float32,
            device=self.cuda_device,
        )
        self.H = torch.fft.fft2(ifftshift2d(self.h_complex))

        self.shutter = np.transpose(shutter, (2, 0, 1))
        self.shutter_var = torch.tensor(
            self.shutter, dtype=torch.float32, device=self.cuda_device
        ).unsqueeze(0)

    def Hfor(self, x):
        xc = torch.complex(x, torch.zeros_like(x))
        X = torch.fft.fft2(xc)

        HX = self.H * X
        out = torch.fft.ifft2(HX)
        out_r = out.real
        return out_r

    def forward(self, in_image):
        output = torch.sum(
            self.shutter_var * crop_forward(self, self.Hfor(my_pad(self, in_image))), 1
        )
        #         output = torch.sum((self.Hfor(my_pad(self, in_image))), 1)

        return output
