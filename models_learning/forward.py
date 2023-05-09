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
        num_ims=2,  # number of blurred images to simulate
        w_init=None,  # initialization for the blur kernels e.g., w_init = [.01, 0.01]
        cuda_device=0,  # cuda device parameter
        blur_type="symmetric",  # symmetric or asymmetric blur kernel
        optimize_blur=False,  # choose whether to learn the best blur or not (warning, not stable)
        simulate_blur=True,
        psf_dir=None,
    ):  # simulate blur or specify psf
        super(Forward_Model, self).__init__()
        self.cuda_device = cuda_device
        self.blur_type = blur_type
        self.num_ims = num_ims
        self.optimize_blur = optimize_blur
        self.simulate_blur = simulate_blur
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
                    0.002, 0.035, num_ims
                )  # sharp bound, blurry bound: (deflt:.002,0.035)
            else:
                w_init = np.linspace(0.002, 0.01, num_ims)
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
                    requires_grad=optimize_blur,
                )
            )
        else:
            self.w_init = w_init
            self.w_blur = torch.nn.Parameter(
                torch.tensor(
                    self.w_init,
                    dtype=torch.float32,
                    device=self.cuda_device,
                    requires_grad=optimize_blur,
                )
            )

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
        self.psf = np.empty((num_ims, self.DIMS0, self.DIMS1))

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
            psfs = psf_utils.get_psf_stack(self.psf_dir)
            assert isinstance(psfs, np.ndarray) or isinstance(
                psfs, torch.Tensor
            ), "psfs must be either np array or tensor"
            assert (
                psfs.shape[0] == self.num_ims
            ), f"must provide {self.num_ims} levels of blur"

            # pad to mask shape
            psfs = psf_utils.center_pad_to_shape(psfs, (self.DIMS0, self.DIMS1), val=0)

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
        sm = pad_zeros_torch(self, sim_meas.unsqueeze(2) * self.mask_var)
        sm_fourier = fft_im(sm)
        adj_meas = torch.fft.ifft2(Hconj * sm_fourier).real
        return adj_meas

    # forward call for the model
    def forward(self, in_image, sim):
        # init psfs if not or dynamic blur
        self.init_psfs()
        if sim:
            self.Xi = fft_im(my_pad(self, in_image)).unsqueeze(0)
            self.sim_meas = self.Hfor()
        else:
            self.sim_meas = in_image
        final_output = crop_forward(self, self.Hadj(self.sim_meas))
        return final_output
