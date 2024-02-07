import matplotlib.pyplot as plt
import torch
import cupy as cp
import numpy as np

import models.fista.tv_approx_haar_cp as cptv
import models.fista.tv_approx_haar_np as nptv
import utils.helper_functions as fc


class fista_spectral_numpy:
    def __init__(self, h, mask, params, device="cpu"):
        self.device = device
        if "cuda" in self.device:
            self.np = cp
            self.tv_lib = cptv

            print(
                "device = ",
                self.np.cuda.Device(int(device[-1])).use(),
                ", using GPU and cupy",
            )
        else:
            self.np = np
            self.tv_lib = nptv
            print("device = ", self.device, ", using CPU and numpy")

        # TODO: Re-implement this with pytorch
        # For now, just convert to numpy/cupy
        if isinstance(h, torch.Tensor):
            h = self.np.array(h.cpu().numpy())
        if isinstance(mask, torch.Tensor):
            mask = self.np.array(mask.cpu().numpy())
        print(type(h), type(mask))

        ## Initialize constants
        self.psfs = h
        self.DIMS0 = h.shape[1]  # Image Dimensions
        self.DIMS1 = h.shape[2]  # Image Dimensions

        self.spectral_channels = mask.shape[-1]  # Number of spectral channels

        self.py = int((self.DIMS0) // 2)  # Pad size
        self.px = int((self.DIMS1) // 2)  # Pad size

        # FFT of point spread function
        self.H = []
        self.Hconj = []
        for i in range(0, h.shape[0]):
            self.H.append(
                self.np.expand_dims(
                    self.np.fft.fft2(
                        (self.np.fft.ifftshift(self.pad(h[i]), axes=(0, 1))),
                        axes=(0, 1),
                    ),
                    -1,
                )
            )
            self.Hconj.append(self.np.conj(self.H[i]))

        self.mask = mask

        # Calculate the eigenvalue to set the step size
        maxeig = self.power_iteration(self.Hpower, (self.DIMS0 * 2, self.DIMS1 * 2), 10)
        self.L = maxeig * 200  # step size for gd update

        self.prox_method = params.get("prox_method", "tv")  # 'non-neg', 'tv', 'native'

        # If using a learned inversion, initialize learned model
        if params.get("learned_inversion", False):
            self.recon_model = None  # TODO implement model instatiation

        # Define soft-thresholding constants
        self.tau = params.get("tau", 0.5)  # Native sparsity tuning parameter
        self.tv_lambda = params.get("tv_lambda", 0.00005)  # TV tuning parameter
        self.tv_lambdaw = params.get("tv_lambdaw", 0.00005)  # TV tuning for wavelength
        self.tv_lambdax = params.get("tv_lambdax", 0.00005)  # TV tuning for wavelength
        self.lowrank_lambda = params.get("lowrank_lambda", 0.00005)  # Low rank tuning
        self.break_diverge_early = False  # Must be manually set

        # Number of iterations of FISTA
        self.iters = params.get("iters", 500)

        self.show_recon_progress = params.get("show_progress", True)  # print progress
        self.print_every = params.get("print_every", 20)  # Frequency to print progress

        self.l_data = []
        self.l_tv = []

    # Power iteration to calculate eigenvalue
    def power_iteration(self, A, sample_vect_shape, num_iters):
        bk = self.np.random.randn(sample_vect_shape[0], sample_vect_shape[1])
        for i in range(0, num_iters):
            bk1 = A(bk)
            bk1_norm = self.np.linalg.norm(bk1)

            bk = bk1 / bk1_norm
        Mx = A(bk)
        xx = self.np.transpose(self.np.dot(bk.ravel(), bk.ravel()))
        eig_b = self.np.transpose(bk.ravel()).dot(Mx.ravel()) / xx

        return eig_b

    # Helper functions for forward model
    def crop(self, x):
        return x[self.py : -self.py, self.px : -self.px]

    def pad(self, x):
        if len(x.shape) == 2:
            out = self.np.pad(
                x, ([self.py, self.py], [self.px, self.px]), mode="constant"
            )
        elif len(x.shape) == 3:
            out = self.np.pad(
                x, ([self.py, self.py], [self.px, self.px], [0, 0]), mode="constant"
            )
        return out

    def Hpower(self, x):
        x = self.np.fft.ifft2(
            self.H[0] * self.np.fft.fft2(self.np.expand_dims(x, -1), axes=(0, 1)),
            axes=(0, 1),
        )
        x = self.np.sum(self.mask * self.crop(self.np.real(x)), 2)
        x = self.pad(x)
        return x

    def Hfor(self, x, i):
        x = self.np.fft.ifft2(self.H[i] * self.np.fft.fft2(x, axes=(0, 1)), axes=(0, 1))
        x = self.np.sum(self.mask * self.crop(self.np.real(x)), 2)
        return x

    def Hadj(self, x, i):
        x = self.np.expand_dims(x, -1)
        x = x * self.mask
        x = self.pad(x)

        x = self.np.fft.fft2(x, axes=(0, 1))
        x = self.np.fft.ifft2(self.Hconj[i] * x, axes=(0, 1))
        x = self.np.real(x)
        return x

    def soft_thresh(self, x, tau):
        out = self.np.maximum(self.np.abs(x) - tau, 0)
        out = out * self.np.sign(x)
        return out

    def prox(self, x):
        if self.prox_method == "tv":
            x = 0.5 * (
                self.np.maximum(x, 0)
                + self.tv_lib.tv3dApproxHaar(
                    x,
                    self.tv_lambda / self.L,
                    self.tv_lambdaw,
                    self.tv_lambdax,
                )
            )
        if self.prox_method == "native":
            x = self.np.maximum(x, 0) + self.soft_thresh(x, self.tau)
        if self.prox_method == "non-neg":
            x = self.np.maximum(x, 0)
        return x

    def tv(self, x):
        d = self.np.zeros_like(x)
        d[0:-1, :] = (x[0:-1, :] - x[1:, :]) ** 2
        d[:, 0:-1] = d[:, 0:-1] + (x[:, 0:-1] - x[:, 1:]) ** 2
        return self.np.sum(self.np.sqrt(d))

    def loss(self, x, err):
        if self.prox_method == "tv":
            self.l_data.append(self.np.linalg.norm(err) ** 2)
            self.l_tv.append(2 * self.tv_lambda / self.L * self.tv(x))

            l = self.np.linalg.norm(err) ** 2 + 2 * self.tv_lambda / self.L * self.tv(x)
        if self.prox_method == "native":
            l = self.np.linalg.norm(
                err
            ) ** 2 + 2 * self.tv_lambda / self.L * self.np.linalg.norm(x.ravel(), 1)
        if self.prox_method == "non-neg":
            l = self.np.linalg.norm(err) ** 2
        return l

    # Main FISTA update
    def fista_update(self, vk, tk, xk, inputs):
        error = 0
        grads = 0
        for i in range(0, inputs.shape[0]):
            #### Learned inversion implementation
            # if learned_inversion:
            #     error = error + self.Hfor(vk, i) - inputs[i]

            #     grads = grads + self.model(error)
            # else:

            ##### old implementation
            # error = error + self.Hfor(vk, i) - inputs[i]
            # grads = grads + self.Hadj(error, i)
            #####

            error_i = self.Hfor(vk, i) - inputs[i]
            grads_i = self.Hadj(error_i, i)

            error += error_i
            grads += grads_i
        error = error / inputs.shape[0]
        grads = grads / inputs.shape[0]

        xup = self.prox(vk - 1 / self.L * grads)
        tup = 1 + self.np.sqrt(1 + 4 * tk**2) / 2
        vup = xup + (tk - 1) / tup * (xup - xk)

        return vup, tup, xup, self.loss(xup, error)

    # Run FISTA
    def run(self, inputs):
        # Initialize variables to zero
        xk = self.np.zeros((self.DIMS0 * 2, self.DIMS1 * 2, self.spectral_channels))
        vk = self.np.zeros((self.DIMS0 * 2, self.DIMS1 * 2, self.spectral_channels))
        tk = 1.0

        llist = []

        # Start FISTA loop
        for i in range(0, self.iters):
            vk, tk, xk, l = self.fista_update(vk, tk, xk, inputs)

            llist.append(l.get())
            if self.break_diverge_early and l.get() > llist[0] * 100:
                print(f"Diverged with loss {l.get():.4f}, stopping...")
                break

            # Print out the intermediate results and the loss
            if self.show_recon_progress == True and i % self.print_every == 0:
                print("iteration: ", i, " loss: ", l)
                if "cuda" in self.device:
                    out_img = self.np.asnumpy(self.crop(xk))
                else:
                    out_img = self.crop(xk)

                if out_img.shape[-1] == 64:
                    fc_img = fc.pre_plot(fc.stack_rgb_opt(out_img))
                else:
                    fc_img = fc.stack_rgb_opt_30(out_img)
                    # fc_img = np.max(out_img, -1)

                plt.figure(figsize=(10, 3), dpi=120)
                plt.subplot(1, 2, 1), plt.imshow(fc_img / np.max(fc_img))
                plt.title("Reconstruction")
                plt.subplot(1, 2, 2), plt.plot(llist)
                plt.title("Loss")
                plt.show()
                self.out_img = out_img
        xout = self.crop(xk)
        self.llist = llist
        return xout, llist

    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            input = self.np.array(input.cpu().numpy())

        assert len(input.shape) in {4, 3}, "Only accepts meas stack, or batch of stacks"

        if len(input.shape) == 4:
            output = np.stack(
                [self.run(input[i, ...])[0] for i in range(input.shape[0])], 0
            )
        else:
            output = self.run(input)[0]

        return output
