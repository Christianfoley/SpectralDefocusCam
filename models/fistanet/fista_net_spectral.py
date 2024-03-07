import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

import models.fista.tv_approx_haar as tv_lib
import utils.helper_functions as helper
from utils.diffuser_utils import *


class fista_net_spectral(torch.nn.Module):
    def __init__(self, h, mask, params, device="cpu"):
        """
        A multi-channel/spectral adaptation of a learned, unrolled fista-net:
            https://doi.org/10.48550/arXiv.2008.02683
            https://github.com/jinxixiang/FISTA-Net

        If "run_fista" set to true, will run an unlearned FISTA with TV.

        Parameters
        ----------
        h : torch.tensor
            PSF stack (n, y, x)
        mask : torch.Tensor
            Spectral calibration mask (c, y, x)
        params : dict
            dictionary of initialization hyperparams, see init_params for more
        device : str, optional
            device to use, by default "cpu"

        Returns
        -------
        torch.nn.Module
            spectral fista-net model
        """
        super(fista_net_spectral, self).__init__()
        self.device = device

        # ----- initialize model parameters ----- #
        self.mask = mask.to(device).to(torch.float32)
        self.psfs = h.to(device).to(torch.float32)

        if not params.get("run_fista", False):
            # fistanet needs energy conserving forward model, so apply l1
            self.psfs = self.psfs / self.psfs.sum(dim=(-2, -1), keepdim=True)
            self.mask = self.mask / self.mask.sum(dim=(-2, -1), keepdim=True)

        # ----- Initialize constants ----- #
        self.dims_c, self.DIMS0, self.DIMS1 = mask.shape[0], h.shape[-2], h.shape[-1]
        self.PAD_SIZE0, self.PAD_SIZE1 = int(self.DIMS0), int(self.DIMS1)
        self.spectral_channels = mask.shape[0]
        self.H = fft_psf(self, self.psfs)
        self.Hconj = torch.conj(self.H)

        # ----- Initialize iteration & fista parameters ----- #
        self.run_fista = params.get("run_fista", False)
        if self.run_fista:
            self.L = (
                self.power_iteration(self.Hpower, (self.DIMS0 * 2, self.DIMS1 * 2), 10)
                * 200
            )
            self.prox_method = params.get("prox_method", "tv")
            self.tv_lambda = params.get("tv_lambda", 0.00005)
            self.tv_lambdaw = params.get("tv_lambdaw", 0.00005)
            self.tv_lambdax = params.get("tv_lambdax", 0.00005)
            self.iters = params.get("iters", 500)
        else:
            self.iters = params.get("iters", 6)
        self.warm_start = params.get("warm_start", True)
        self.show_recon_progress = params.get("show_progress", True)
        self.print_every = params.get("print_every", 20)

        # ----- initialize learned parameters -----#
        self.conv3d = params.get("conv3d", False)
        self.n_features = params.get("num_features", 48)
        if self.conv3d:
            self.forward_block = forward_block3d(self.n_features)
            self.backward_block = backward_block3d(self.n_features)
        else:
            self.forward_block = forward_block(self.dims_c, self.n_features)
            self.backward_block = backward_block(self.dims_c, self.n_features)
        self.forward_block.apply(initialize_weights)
        self.backward_block.apply(initialize_weights)
        self.Sp = nn.Softplus()

        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([params.get("w_theta", -0.5)]))
        self.b_theta = nn.Parameter(torch.Tensor([params.get("b_theta", -10)]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([params.get("w_mu", -0.2)]))
        self.b_mu = nn.Parameter(torch.Tensor([params.get("b_mu", 0.1)]))
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([params.get("w_rho", 0.5)]))
        self.b_rho = nn.Parameter(torch.Tensor([params.get("b_rho", 0)]))

    # Power iteration to calculate eigenvalue for stepsize estimation
    def power_iteration(self, A, sample_vect_shape, num_iters):
        """
        Power iteration utility: finds max eigenvalue for fista step size
        """
        bk = torch.randn(sample_vect_shape[0], sample_vect_shape[1], device=self.device)
        for i in range(0, num_iters):
            bk1 = A(bk)
            bk1_norm = torch.linalg.norm(bk1)

            bk = bk1 / bk1_norm
        Mx = A(bk)

        def torch_transpose(A):
            dims = tuple(reversed(range(A.dim())))
            return torch.permute(A, dims=dims)

        xx = torch_transpose(torch.dot(bk.ravel(), bk.ravel()))
        eig_b = torch_transpose(bk.ravel()).dot(Mx.ravel()) / xx

        return eig_b

    def Hpower(self, x):
        """
        Version of forward model using only the first PSF as a utility function
        """
        x = torch.fft.ifft2(
            self.H[0] * torch.fft.fft2(x, dim=(-2, -1))[None, ...], dim=(-2, -1)
        )
        x = torch.sum(self.mask * crop_forward(self, torch.real(x)), 0)
        x = pad_zeros_torch(self, x)
        return x

    # Physical model simulation methods
    def Hfor(self, x):
        """
        LSI spatially invariant system forward method

        Parameters
        ----------
        x : torch.Tensor
            ground truth / estimated hyperspectral batch (c, y, x)

        Returns
        -------
        torch.Tensor
            simulated measurements (n, y, x)
        """
        X = torch.fft.fft2(x, dim=(-2, -1))
        x = torch.fft.ifft2(self.H * X, dim=(-2, -1)).to(torch.float32)
        x = torch.sum(self.mask * crop_forward(self, torch.real(x)), -3)
        return x

    def Hadj(self, x):
        """
        LSI spatially invariant system adjoint method

        Parameters
        ----------
        x : torch.Tensor
            ground truth / estimated hyperspectral batch (n, y, x)

        Returns
        -------
        torch.Tensor
            simulated measurements (n, c, y, x)
        """
        x = torch.unsqueeze(x, -3)
        x = x * self.mask
        x = pad_zeros_torch(self, x)

        x = torch.fft.fft2(x, dim=(-2, -1))
        x = torch.fft.ifft2(self.Hconj * x, dim=(-2, -1)).to(torch.float32)
        x = torch.real(x)
        return x

    # Thresholding utils
    def soft_thresh(self, x, tau):
        """
        Soft thresholding from a threshold Tau
        """
        out = torch.mul(torch.sign(x), F.relu(torch.abs(x) - tau))
        return out

    def soft_thresh_tv(self, x, tau=None):
        """
        Soft thresholding with TV (gradient sparsity constraint)
        """
        thresh = self.tv_lambda / self.L
        if tau is not None:
            thresh = tau

        x = 0.5 * (
            F.relu(x)
            + tv_lib.tv3dApproxHaar(x, thresh, self.tv_lambdaw, self.tv_lambdax)
        )
        return x

    # model building blocks
    def model_layer(self, v_k, measurements, lambda_step):
        """
        Physics-guided model module. Computes the gradients & next measurement
        estimate as in the estimation step of a fista iteration:
            https://doi.org/10.48550/arXiv.2008.02683
            eq (8a)

        Parameters
        ----------
        v_k : torch.Tensor
            measurement estimate (c, y, x)
        measurements : torch.Tensor
            signal measurement stack (n, y, x)
        lambda_step : torch.Tensor
            learning rate/step parameter
        """
        error = self.Hfor(v_k) - measurements
        grads = self.Hadj(error)

        updated_est = v_k - lambda_step * torch.mean(grads, -4)

        return updated_est

    def learned_layer(self, x_input, t_soft):
        """
        Forward pass through the learned layer, computes the measurement estimate
        as in the inversion step of a fista interation:
            https://doi.org/10.48550/arXiv.2008.02683
            eq (8c)

        Parameters
        ----------
        x_input : torch.Tensor
            input to the layer (n, c, y, x)
        t_soft : torch.Tensor
            soft threshold
        """
        # fwd pass through layer
        x_for, x_D = self.forward_block(x_input)
        plt.imshow(np.mean(x_for[0].detach().cpu().numpy(), 0))
        plt.colorbar()
        plt.show()
        x_for_st = self.soft_thresh(x_for, t_soft)
        plt.imshow(np.mean(x_for_st[0].detach().cpu().numpy(), 0))
        plt.colorbar()
        plt.show()
        _, x_G = self.backward_block(x_for_st)

        plt.imshow(np.mean(x_G[0].detach().cpu().numpy(), 0))
        plt.colorbar()
        plt.show()

        # enforce symmetry
        x_D_est, _ = self.backward_block(x_for)
        symloss = x_D_est - x_D

        # prediction from layer (non-negative residual/skip connection)
        pred = F.relu(x_input + x_G)

        return pred, symloss, x_for_st

    def update_hparams(self, i):
        """
        Recalculate hyperparameters for this layer from learned variables

        Parameters
        ----------
        i : int
            iteration/layer number
        """
        theta = self.Sp(self.w_theta * i + self.b_theta)
        mu = self.Sp(self.w_mu * i + self.b_mu)
        rho = (self.Sp(self.w_rho * i + self.b_rho) - self.Sp(self.b_rho)) / self.Sp(
            self.w_rho * i + self.b_rho
        )

        return theta, mu, rho

    def init_estimates(self, inputs, size):
        """
        Initialize estimate intermediate variables. If warm starting is used, will
        initialize to the adjoint of the measurements. Otherwise will initialize to
        zero.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched stack of input measurements (b, n, y, x)
        size : tuple
            size of variables typically (c, y*2, x*2) or (c, y, x)

        Returns
        -------
        _type_
            _description_
        """
        xk = torch.zeros(inputs.shape[0:1] + size).to(self.device).to(torch.float32)
        vk = torch.zeros(inputs.shape[0:1] + size).to(self.device).to(torch.float32)
        tk = torch.tensor(1.0).to(torch.float32)

        if self.warm_start:
            xk = torch.mean(self.Hadj(inputs), dim=1)
            vk = torch.mean(self.Hadj(inputs), dim=1)

        return vk, xk, tk

    # Main updates
    def fista_net_update(self, vk, xk, b, t_soft, lambda_step, rho):
        """
        Run a single pass through the the network, as in a single iteration of fista

        Parameters
        ----------
        vk : torch.Tensor
            intermediate variable (initialized at k=0 to laplacian reg)
        xk : torch.Tensor
            estimate at step k (n, c, y, x)
        b : torch.Tensor
            stack of signal measurements (n, y, x)
        t_soft : torch.Tensor
            learned soft threshold param
        lambda_step : torch.Tensor
            learned gradient descent step param
        rho : torch.Tensor
            learned two-step update step param

        Returns
        -------
        tuple
            updated intermediate, estimate, and error tensors
        """
        # run network layer update
        x_up = self.model_layer(vk, b, lambda_step)
        plt.imshow(np.mean(x_up[0].detach().cpu().numpy(), 0))
        plt.colorbar()
        plt.show()
        x_up, layer_sym, layer_st = self.learned_layer(x_up, t_soft)
        v_up = x_up + rho * (x_up - xk)  # two-step update

        return v_up, x_up, layer_sym, layer_st

    def fista_update(self, vk, tk, xk, inputs):
        """
        Fista update to test fista implementation components
        """
        x_up = self.model_layer(vk, inputs, 1 / self.L)
        x_up = self.soft_thresh_tv(x_up)

        t_up = 1 + torch.sqrt(1 + 4 * tk**2) / 2
        v_up = x_up + (tk - 1) / t_up * (x_up - xk)

        torch.cuda.empty_cache()
        return v_up, t_up, x_up

    # Run FISTA
    def run(self, inputs):
        """
        Runs the forward method of fistanet.
        Note: if self.run_fista is set to True, will run an unlearned fista with tv
        recon. If set to false, will run the learned recon.

        Parameters
        ----------
        inputs : torch.Tensor
            Batched stack of input measurements (b, n, y, x)

        Returns
        -------
        torch.Tensor
            Batched reconstructed hyperspectral volume (b, c, y, x)
        list[torch.Tensor]
            Symmetry loss elements (if running fistanet)
        list[torch.Tensor]
            Sparsity loss elements (if running fistanet)
        """
        # Initialize variables
        if self.run_fista:
            inputs = helper.value_norm(inputs)
        var_size = (self.spectral_channels, self.DIMS0 * 2, self.DIMS1 * 2)
        vk, xk, tk = self.init_estimates(inputs, var_size)
        layer_loss_sym, layer_loss_st = [], []

        # Start iterative reconstruction
        for i in range(0, self.iters):
            if self.run_fista:
                vk, tk, xk = self.fista_update(vk, tk, xk, inputs)
            else:
                t_soft, lambda_step, rho = self.update_hparams(i)

                vk, xk, layer_sym, layer_st = self.fista_net_update(
                    vk, xk, inputs, t_soft, lambda_step, rho
                )

                layer_loss_sym.append(layer_sym)
                layer_loss_st.append(layer_st)

            # show intermediate results at print stride and last iteration
            if self.show_recon_progress == True and (
                i % self.print_every == 0 or i == self.iters - 1
            ):
                self.plot_progress(crop_forward(self, xk), i)

        xout = crop_forward(self, xk)
        if self.run_fista:
            return xout
        else:
            return xout, layer_loss_sym, layer_loss_st

    def plot_progress(self, xk, i):
        """
        Helper function for plotting current reconstruction output
        """

        def preplot(data):
            out_img = helper.stack_rgb_opt_30(np.transpose(data, (1, 2, 0)))
            return helper.value_norm(out_img)

        self.out_img = xk.detach().cpu().numpy()
        num_plots = self.out_img.shape[0]

        if num_plots == 1:
            plt.figure(figsize=(5, 5), dpi=100)
            plt.imshow(preplot(self.out_img[0]))
            plt.title(f"Reconstruction {i}")
            plt.show()
        else:
            fig, ax = plt.subplots(1, num_plots, figsize=(num_plots * 5, 5), dpi=100)
            for j in range(num_plots):
                ax[j].imshow(preplot(self.out_img[i]))
                ax[j].set_title(f"Batchnum {j}")
            plt.suptitle(f"Reconstructions {i}")

            plt.show()

    def forward(self, input):
        if len(input.shape) == 5:
            input = torch.squeeze(input, -3)  # squeeze meas stack channel dim
        elif len(input.shape) == 3:
            input = input.unsqueeze(0)
        assert (
            len(input.shape) == 4
        ), "Only accepts batch of measurement stacks (b,n,y,x)"

        output = self.run(input.to(torch.float32))
        return output


class forward_block(nn.Module):
    def __init__(self, n, features=48):
        """
        Forward component of learned proximal layer

        Parameters
        ----------
        n : int
            number of measurements
        features : int, optional
            number of latent features, by default 16
        """
        super(forward_block, self).__init__()
        self.Sp = nn.Softplus()

        self.conv_D = nn.Conv2d(n, features, 3, stride=1, padding=1)
        self.conv1_forward = nn.Conv2d(features, features, 3, stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(features, features, 3, stride=1, padding=1)
        # self.conv3_forward = nn.Conv2d(features, features, 3, stride=1, padding=1)
        # self.conv4_forward = nn.Conv2d(features, features, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            input estimate to encode (b, 1, c, y, x) or (b, c, y, x)
        """
        if len(x.shape) == 5:
            x = x.squeeze(1)

        x = x.to(torch.float32)
        x_D = self.conv_D(x)

        x = self.conv1_forward(x_D)
        x = F.relu(x)
        x = self.conv2_forward(x)
        # x = F.relu(x)
        # x = self.conv3_forward(x)
        # x = F.relu(x)
        # x = self.conv4_forward(x)

        return x, x_D


class backward_block(nn.Module):
    def __init__(self, n, features=16):
        """
        Backward component of learned proximal layer

        Parameters
        ----------
        n : int
            number of measurements
        features : int, optional
            number of latent features, by default 32
        """
        super(backward_block, self).__init__()
        self.Sp = nn.Softplus()

        self.conv1_backward = nn.Conv2d(features, features, 3, stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(features, features, 3, stride=1, padding=1)
        # self.conv3_backward = nn.Conv2d(features, features, 3, stride=1, padding=1)
        # self.conv4_backward = nn.Conv2d(features, features, 3, stride=1, padding=1)
        self.conv_G = nn.Conv2d(features, n, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            input estimate to decode (b, 1, c, y, x) or (b, c, y, x)
        """
        unsqueeze = False
        if len(x.shape) == 5:
            unsqueeze = True
            x = x.squeeze(1)

        x = x.to(torch.float32)
        x = self.conv1_backward(x)
        x = F.relu(x)
        x = self.conv2_backward(x)
        # x = F.relu(x)
        # x = self.conv3_backward(x)
        # x = F.relu(x)
        # x = self.conv4_backward(x)

        x_G = self.conv_G(x)

        if unsqueeze:
            return x[:, None, ...], x_G[:, None, ...]
        else:
            return x, x_G


class forward_block3d(nn.Module):
    def __init__(self, features=16):
        """
        Forward component of learned proximal layer

        Parameters
        ----------
        features : int, optional
            number of latent features, by default 16
        """
        super(forward_block3d, self).__init__()
        self.Sp = nn.Softplus()

        self.conv_D = nn.Conv3d(1, features, 3, stride=1, padding=1)
        self.conv1_forward = nn.Conv3d(features, features, 3, stride=1, padding=1)
        self.conv2_forward = nn.Conv3d(features, features, 3, stride=1, padding=1)
        self.conv3_forward = nn.Conv3d(features, features, 3, stride=1, padding=1)
        self.conv4_forward = nn.Conv3d(features, features, 3, stride=1, padding=1)

    def forward(self, x):
        x = x.to(torch.float32)
        x_D = self.conv_D(x)

        x = self.conv1_forward(x_D)
        x = F.relu(x)
        x = self.conv2_forward(x)
        x = F.relu(x)
        x = self.conv3_forward(x)
        x = F.relu(x)
        x = self.conv4_forward(x)

        return x, x_D


class backward_block3d(nn.Module):
    def __init__(self, features=16):
        """
        Backward component of learned proximal layer

        Parameters
        ----------
        n : int
            number of measurements
        features : int, optional
            number of latent features, by default 32
        """
        super(backward_block3d, self).__init__()
        self.Sp = nn.Softplus()

        self.conv1_backward = nn.Conv3d(features, features, 3, stride=1, padding=1)
        self.conv2_backward = nn.Conv3d(features, features, 3, stride=1, padding=1)
        self.conv3_backward = nn.Conv3d(features, features, 3, stride=1, padding=1)
        self.conv4_backward = nn.Conv3d(features, features, 3, stride=1, padding=1)
        self.conv_G = nn.Conv3d(features, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv1_backward(x)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x = self.conv4_backward(x)

        x_G = self.conv_G(x)

        return x, x_G


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)
