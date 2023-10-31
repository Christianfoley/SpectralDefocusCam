import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os, glob, cv2

import data_utils.dataset as ds
from utils.diffuser_utils import *
import utils.psf_calibration_utils as psf_utils

import models.rdmpy.blur as blur


class Forward_Model(torch.nn.Module):
    def __init__(
        self,
        mask,
        params,
        psf_dir=None,
        w_init=None,
        device=torch.device("cpu"),
    ):
        """
        Initializes Forward_Model() object, used for forward physics simulation of
        imaging system.

        For details pertaining to the physics model, see precompute_data_lri.ipynb.
        For base (default) parameters see configs/base_config.py

        Parameters
        ----------
        mask : torch.Tensor
            spectral response (transmittance) matrix for the spectral filter array
        params : dict
            dictionary of params, must contain {'stack_depth', 'operations', 'psf'}
        psf_dir : str
            path to directory containing LRI or LSI psf data, by default None
        w_init : list, optional
            specific initialization for the blur kernels, by default None
        device : torch.Device
            device to perform operations & hold tensors on, by default "cpu"
        """
        super(Forward_Model, self).__init__()
        self.device = device
        self.psf_dir, self.psfs = psf_dir, None
        self.num_ims = params["stack_depth"]

        # Which forward operations to perform
        self.operations = params["operations"]

        # what psfs to use/simulate
        self.psf = params["psf"]

        ## Initialize constants
        self.DIMS0 = mask.shape[0]  # Image Dimensions
        self.DIMS1 = mask.shape[1]  # Image Dimensions
        self.PAD_SIZE0 = int((self.DIMS0))  # Pad size
        self.PAD_SIZE1 = int((self.DIMS1))  # Pad size

        if w_init is None:
            self.w_init = np.linspace(0.002, 0.035, self.num_ims)
            if not self.psf["symmetric"]:
                self.w_init = np.linspace(0.002, 0.01, self.num_ims)
                self.w_init = np.repeat(
                    np.array(self.w_init)[np.newaxis], self.num_ims, axis=0
                ).T
                self.w_init[:, 1] *= 0.5

        self.w_blur = torch.nn.Parameter(
            torch.tensor(
                self.w_init,
                dtype=torch.float32,
                device=self.device,
                requires_grad=self.optimize_blur,
            ),
            requires_grad=self.optimize_blur,
        )

        # set up grid
        x = np.linspace(-1, 1, self.DIMS1)
        y = np.linspace(-1, 1, self.DIMS0)
        X, Y = np.meshgrid(x, y)

        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.Y = torch.tensor(Y, dtype=torch.float32, device=self.device)

        self.mask = np.transpose(mask, (2, 0, 1))
        self.mask_var = torch.tensor(
            self.mask, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # determine forward and adjoint methods
        self.fwd = self.Hfor
        self.adj = self.Hadj
        if self.psf["lri"]:
            self.fwd = self.Hfor_varying
            self.adj = self.Hadj_varying

    def simulate_lsi_psf(self):
        """
        Simulates gaussian-distributed PSFS for an LSI system. Initial psf variance
        specified by self.w_init

        Returns
        -------
        torch.Tensor
            stack of increasingly defocused psfs (n, y, x)
        """
        psfs = []
        for i in range(0, self.num_ims):
            var_yx = (self.w_blur[i], self.w_blur[i])
            if not self.psf["symmetric"]:
                var_yx = (self.w_blur[i, 0], self.w_blur[i, 1])

            psf = torch.exp(-((self.X / var_yx[0]) ** 2 + (self.Y / var_yx[1]) ** 2))
            psfs.append(psf / torch.linalg.norm(psf, ord=float("inf")))
        return torch.stack(psfs, 0)

    def init_psfs(self):
        """
        Initializes psfs depending on the specified parameters.
            - If simulating psfs (LSI only), generates initial psf values
            - If using measured LSI psfs, loads and preprocesses psf stack
            - If using LRI psfs, loads in calibrated psf data
        """
        if self.psf["lri"]:
            psf_utils.read_psfs(self.psf_dir)

            return

        if self.operations["sim_blur"]:
            psfs = self.simulate_LSI_psf()
        elif self.psfs == None:
            psfs = psf_utils.get_psf_stack(
                self.psf_dir,
                self.num_ims,
                self.mask.shape,
                self.psf["padded_shape"],
                one_norm=self.psf["norm_each"],
                blurstride=self.psf["stride"],
            )
        self.psfs = torch.tensor(psfs, dtype=torch.float32, device=self.device)

    def Hfor(self, x):
        """
        LSI spatially invariant system forward method

        Parameters
        ----------
        v : torch.Tensor
            ground truth hyperspectral batch (b, 1, c, y, x)
        h : torch.Tensor
            measured or simulated psf data (n, y, x)
        mask : torch.Tensor
            spectral calibration matrix/mask (c, y, x)

        Returns
        -------
        torch.Tensor
            simulated measurements (b, n, 1, y, x)
        """
        X = fft_im(my_pad(self, x)).unsqueeze(2)
        H = fft_psf(self, self.psfs)
        out = torch.fft.ifft2(H * X).real
        output = torch.sum(self.mask_var * crop_forward(self, out), 2)
        return output

    def Hadj(self, sim_meas):
        """
        LSI spatially invariant convolution adjoint method

        Parameters
        ----------
        b : torch.Tensor
            simulated (or not) measurement (b, n, 1, y, x)
        h : torch.Tensor
            measured or simulated psf data (n,y,x)
        mask : torch.Tensor
            spectral calibration matrix/mask (c, y, x)

        Returns
        -------
        torch.Tensor
            LSI system adjoint
        """
        Hconj = torch.conj(fft_psf(self, self.psfs))
        sm = pad_zeros_torch(self, sim_meas * self.mask_var)
        adj_meas = torch.fft.ifft2(Hconj * fft_im(sm)).real
        return adj_meas

    def Hfor_varying(self, v, h, mask):
        """
        LRI spatially varying system forward method

        Parameters
        ----------
        v : torch.Tensor
            ground truth hyperspectral batch (b, 1, c, y, x)
        h : torch.Tensor
            calibrated psf data (n, y, theta, x)
        mask : torch.Tensor
            spectral calibration matrix/mask (c, y, x)

        Returns
        -------
        torch.Tensor
            simulated measurements (b, n, 1, y, x)
        """
        b = torch.sum(mask * crop_forward(batch_ring_convolve(v, h, self.device)), 2)
        return b

    def Hadj_varying(self, b, h, mask):
        """
        LRI spatially varying convolution adjoint method

        Parameters
        ----------
        b : torch.Tensor
            simulated (or not) measurement (b, n, 1, y, x)
        h : torch.Tensor
            calibrated psf data (n, y, theta, x)
        mask : torch.Tensor
            spectral calibration matrix/mask (c, y, x)

        Returns
        -------
        torch.Tensor
            LRI system adjoint (b, n, c, y, x)
        """
        v_hat = batch_ring_convolve(
            pad_zeros_torch(self, b * mask), torch.conj(h), self.device
        ).real
        return v_hat

    def spectral_pad(self, x, spec_dim=2, size=-1):
        """
        Zero pads spectral dimension of input tensor x to a power of 2

        Parameters
        ----------
        x : torch.Tensor
            input tensor to be padded (b, n, c, y, x)
        spec_dim : int, optional
            index of spectral dimension in tensor, by default 2
        size : int, optional
            size of padding, if specific size is requested, by default -1

        Returns
        -------
        torch.Tensor
            padded output tensor
        """
        spec_channels = x.shape[spec_dim]
        padsize = 0
        while spec_channels & (spec_channels - 1) != 0:
            spec_channels += 1
            padsize += 1
        padsize = size if size >= 0 else padsize
        return F.pad(
            x, (0, 0, 0, 0, padsize // 2, padsize // 2 + padsize % 2), "constant", 0
        )

    def forward(self, in_image):
        """
        Applies forward model operations, as specified in initialization

        Parameters
        ----------
        in_image : torch.tensor
            5d tensor (b, n, c, y, x)

        Returns
        -------
        torch.Tensor
            5d tensor (b, n, 1, y x)
        """
        self.init_psfs()

        if self.operations["sim_meas"]:
            self.sim_output = self.fwd(in_image)
        else:
            self.sim_output = in_image

        if self.operations["adjoint"]:
            self.sim_output = crop_forward(self, self.adj(self.sim_output))

        # spawn a channel dimension and apply a global normalization
        self.sim_output = self.sim_output.unsqueeze(2)
        self.sim_output = (self.sim_output - torch.min(self.sim_output)) / torch.max(
            self.sim_output - torch.min(self.sim_output)
        )

        if self.operations["spectral_pad"]:
            self.sim_output = self.spectral_pad(self.sim_output, spec_dim=2, size=2)

        if self.operations["roll"]:
            self.sim_output = torch.roll(self.sim_output, self.num_ims // 2, dims=1)

        return self.sim_output
