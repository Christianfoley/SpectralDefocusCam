import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os, glob, cv2
from utils.diffuser_utils import *
import data_utils.dataset as ds
import utils.psf_calibration_utils as psf_utils


class Forward_Model(torch.nn.Module):
    # Model initialization
    def __init__(
        self,
        mask,  # mask: response matrix for the spectral filter array
        params,  # hyperparams related to forward model funciton (num ims, channels, padding, etc)
        w_init=None,  # initialization for the blur kernels e.g., w_init = [.01, 0.01]
        cuda_device=0,  # cuda device parameter
        psf_dir=None,
    ):  # simulate blur or specify psf
        super(Forward_Model, self).__init__()
        self.num_ims = params["stack_depth"]
        self.blur_type = params["blur_type"]
        self.optimize_blur = params["optimize_blur"]
        self.simulate_blur = params["sim_blur"]
        self.simulate_meas = params["sim_meas"]
        self.apply_adjoint = params["apply_adjoint"]
        self.pad = params["spectral_pad_output"]

        self.cuda_device = cuda_device
        self.psf_dir = psf_dir
        self.psfs = None

        ## Initialize constants
        self.DIMS0 = mask.shape[0]  # Image Dimensions
        self.DIMS1 = mask.shape[1]  # Image Dimensions
        self.PAD_SIZE0 = int((self.DIMS0))  # Pad size
        self.PAD_SIZE1 = int((self.DIMS1))  # Pad size

        if w_init is None:  # if no blur specified, use default
            if self.blur_type == "symmetric":
                w_init = np.linspace(
                    0.002, 0.035, self.num_ims
                )  # sharp bound, blurry bound: (deflt:.002,0.035)
            else:
                w_init = np.linspace(0.002, 0.01, self.num_ims)
                w_init = np.repeat(np.array(w_init)[np.newaxis], self.num_ims, axis=0).T
                w_init[:, 1] *= 0.5

        if self.blur_type == "symmetric":
            # self.w_init =  np.repeat(np.array(w_init[0])[np.newaxis], self.num_ims, axis = 0)
            self.w_init = w_init
            self.w_blur = torch.nn.Parameter(
                torch.tensor(
                    self.w_init,
                    dtype=torch.float32,
                    device=self.cuda_device,
                )
            )
            if not self.optimize_blur:
                self.w_blur.requires_grad = False
        else:
            self.w_init = w_init
            self.w_blur = torch.nn.Parameter(
                torch.tensor(
                    self.w_init,
                    dtype=torch.float32,
                    device=self.cuda_device,
                )
            )
            if not self.optimize_blur:
                self.w_blur.requires_grad = False

        # set up grid
        x = np.linspace(-1, 1, self.DIMS1)
        y = np.linspace(-1, 1, self.DIMS0)
        X, Y = np.meshgrid(x, y)

        self.X = torch.tensor(X, dtype=torch.float32, device=self.cuda_device)
        self.Y = torch.tensor(Y, dtype=torch.float32, device=self.cuda_device)

        self.mask = np.transpose(mask, (2, 0, 1))
        self.mask_var = torch.tensor(
            self.mask, dtype=torch.float32, device=self.cuda_device
        ).unsqueeze(0)

    def init_psfs(self):
        if self.simulate_blur:
            psfs = []
            for i in range(0, self.num_ims):
                if self.blur_type == "symmetric":
                    psf = torch.exp(
                        -(
                            (self.X / self.w_blur[i]) ** 2
                            + (self.Y / self.w_blur[i]) ** 2
                        )
                    )
                else:
                    psf = torch.exp(
                        -(
                            (self.X / self.w_blur[i, 0]) ** 2
                            + (self.Y / self.w_blur[i, 1]) ** 2
                        )
                    )
                psf = psf / torch.linalg.norm(psf, ord=float("inf"))
                psfs.append(psf)
            self.psfs = torch.stack(psfs, 0)
        elif self.psfs == None:
            psfs = psf_utils.get_psf_stack(self.psf_dir, self.num_ims, self.mask.shape)

            # convert back to tensor
            if isinstance(psfs, np.ndarray):
                psfs = torch.tensor(psfs, dtype=torch.float32, device=self.cuda_device)

            self.psfs = psfs

    def Hfor(self):
        H = fft_psf(self, self.psfs)
        X = self.Xi.unsqueeze(2)[0]
        out = torch.fft.ifft2(H * X).real
        output = self.mask_var * crop_forward(self, out)
        output = torch.sum(output, 2)

        return output

    def Hadj(self, sim_meas):
        Hconj = torch.conj(fft_psf(self, self.psfs))
        sm = pad_zeros_torch(self, sim_meas * self.mask_var)
        sm_fourier = fft_im(sm)
        adj_meas = torch.fft.ifft2(Hconj * sm_fourier).real
        return adj_meas

    def spectral_pad(self, x, spec_dim=2, size=-1):
        spec_channels = x.shape[spec_dim]
        padsize = 0
        while spec_channels & (spec_channels - 1) != 0:
            spec_channels += 1
            padsize += 1
        padsize = size if size >= 0 else padsize
        return F.pad(
            x, (0, 0, 0, 0, padsize // 2, padsize // 2 + padsize % 2), "constant", 0
        )

    # forward call for the model
    def forward(self, in_image):
        self.init_psfs()

        # simulate camera measurements (or not)
        if self.simulate_meas:
            self.Xi = fft_im(my_pad(self, in_image)).unsqueeze(0)
            self.sim_output = self.Hfor()
        else:
            self.sim_output = in_image

        # spawn a channel dimension
        self.sim_output = self.sim_output.unsqueeze(2)

        # apply the adjoint (or not)
        if self.apply_adjoint:
            self.sim_output = crop_forward(self, self.Hadj(self.sim_output))

        # pad to power of two if specified
        if self.pad:
            self.sim_output = self.spectral_pad(self.sim_output, spec_dim=2, size=2)

        # roll to fix axis switching
        self.sim_output = torch.roll(self.sim_output, self.num_ims // 2, dims=1)
        return self.sim_output
