import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

import models.fista.tv_approx_haar as tv_lib
import utils.helper_functions as fc


class fista_spectral(torch.nn.Module):
    def __init__(self, h, mask, params, learned_recon_model=None, device="cpu"):
        self.device = device
        print(self.device)

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
                torch.unsqueeze(
                    torch.fft.fft2(
                        (torch.fft.ifftshift(self.pad(h[i]), dim=(0, 1))),
                        dim=(0, 1),
                    ),
                    -1,
                ).to(self.device)
            )
            self.Hconj.append(torch.conj(self.H[i]).to(self.device))

        self.mask = mask.to(device)

        # Calculate the eigenvalue to set the step size
        maxeig = self.power_iteration(self.Hpower, (self.DIMS0 * 2, self.DIMS1 * 2), 10)
        self.L = maxeig * 200  # step size for gd update

        self.prox_method = params.get("prox_method", "tv")  # 'non-neg', 'tv', 'native'

        # If using a learned inversion, initialize learned model
        self.learned_inversion = False
        if learned_recon_model is not None:
            self.learned_inversion = True
            self.recon_model = learned_recon_model

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

    # Helper functions for forward model
    def crop(self, x):
        return x[self.py : -self.py, self.px : -self.px]

    def pad(self, x):
        if len(x.shape) == 2:
            out = F.pad(x, (self.px, self.px, self.py, self.py)[::-1], mode="constant")
        elif len(x.shape) == 3:
            out = F.pad(x, (0, 0, self.px, self.px, self.py, self.py), mode="constant")
        return out

    def Hpower(self, x):
        x = torch.fft.ifft2(
            self.H[0] * torch.fft.fft2(torch.unsqueeze(x, -1), dim=(0, 1)),
            dim=(0, 1),
        )
        x = torch.sum(self.mask * self.crop(torch.real(x)), 2)
        x = self.pad(x)
        return x

    def Hfor(self, x, i):
        x = torch.fft.ifft2(self.H[i] * torch.fft.fft2(x, dim=(0, 1)), dim=(0, 1))
        x = torch.sum(self.mask * self.crop(torch.real(x)), 2)
        return x

    def Hadj(self, x, i):
        x = torch.unsqueeze(x, -1)
        x = x * self.mask
        x = self.pad(x)

        x = torch.fft.fft2(x, dim=(0, 1))
        x = torch.fft.ifft2(self.Hconj[i] * x, dim=(0, 1))
        x = torch.real(x)
        return x

    def soft_thresh(self, x, tau):
        out = torch.maximum(torch.abs(x) - tau, torch.tensor(0))
        out = out * torch.sign(x)
        return out

    def prox(self, x):
        if self.prox_method == "tv":
            x = 0.5 * (
                torch.maximum(x, torch.tensor(0))
                + tv_lib.tv3dApproxHaar(
                    x,
                    self.tv_lambda / self.L,
                    self.tv_lambdaw,
                    self.tv_lambdax,
                )
            )
        if self.prox_method == "native":
            x = torch.maximum(x, torch.tensor(0)) + self.soft_thresh(x, self.tau)
        if self.prox_method == "non-neg":
            x = torch.maximum(x, torch.tensor(0))
        return x

    def tv(self, x):
        d = torch.zeros_like(x)
        d[0:-1, :] = (x[0:-1, :] - x[1:, :]) ** 2
        d[:, 0:-1] = d[:, 0:-1] + (x[:, 0:-1] - x[:, 1:]) ** 2
        return torch.sum(torch.sqrt(d))

    def loss(self, x, err):
        if self.prox_method == "tv":
            self.l_data.append(torch.linalg.norm(err) ** 2)
            self.l_tv.append(2 * self.tv_lambda / self.L * self.tv(x))

            l = torch.linalg.norm(err) ** 2 + 2 * self.tv_lambda / self.L * self.tv(x)
        if self.prox_method == "native":
            l = torch.linalg.norm(
                err
            ) ** 2 + 2 * self.tv_lambda / self.L * torch.linalg.norm(x.ravel(), 1)
        if self.prox_method == "non-neg":
            l = torch.linalg.norm(err) ** 2
        return l

    def lrnd_recon(self, x):
        x = torch.stack([self.crop(self.Hadj(x[i], i)) for i in range(x.shape[0])])
        x = torch.transpose(torch.transpose(x, -1, -3), -2, -1)[None, ...].float()
        x = F.pad(x, (0, 0, 0, 0, 1, 1), mode="constant", value=0.0)

        mean, std = x.mean(), x.std()
        norm = lambda y: (y - mean) / (std + torch.tensor(2e-10))
        denorm = lambda y: (y + mean) * std

        x_inv = self.recon_model(norm(x)).detach()[:, 1:-1, ...]
        x_inv = denorm(torch.transpose(torch.transpose(x_inv[0], 0, 2), 0, 1))

        return self.pad(x_inv)

    def clip(self, xk, percentile=0.99):
        v = torch.quantile(xk, percentile)
        return torch.clamp(xk, None, v)

    # Main FISTA update
    def fista_update(self, vk, tk, xk, inputs):
        error = torch.zeros(self.DIMS0, self.DIMS1, device=self.device)
        grads = torch.zeros_like(vk, device=self.device)

        if self.learned_inversion:
            estimates = torch.zeros_like(inputs, device=self.device)

        for i in range(0, inputs.shape[0]):
            #### Learned inversion implementation
            if self.learned_inversion:
                estimates[i] = self.Hfor(vk, i)
                error_i = estimates[i] - inputs[i]
                error += error_i
            else:
                error = error + self.Hfor(vk, i) - inputs[i]
                grads = grads + self.Hadj(error, i)

        if self.learned_inversion:
            sim_recon = self.lrnd_recon(estimates)
            meas_recon = self.lrnd_recon(inputs)
            grads = sim_recon - meas_recon
            del meas_recon, sim_recon

        error = error / inputs.shape[0]
        grads = grads / inputs.shape[0]

        # xup = self.clip(self.prox(vk - 1 / self.L * grads)) #
        xup = self.prox(vk - 1 / self.L * grads)
        tup = 1 + torch.sqrt(1 + 4 * tk**2) / 2
        vup = xup + (tk - 1) / tup * (xup - xk)

        del grads
        torch.cuda.empty_cache()

        return vup, tup, xup, self.loss(xup, error)

    # Run FISTA
    def run(self, inputs):
        # Initialize variables to zero
        xk = torch.zeros((self.DIMS0 * 2, self.DIMS1 * 2, self.spectral_channels)).to(
            self.device
        )
        vk = torch.zeros((self.DIMS0 * 2, self.DIMS1 * 2, self.spectral_channels)).to(
            self.device
        )
        tk = torch.tensor(1.0)

        llist = []

        # Start FISTA loop
        for i in range(0, self.iters):
            vk, tk, xk, l = self.fista_update(vk, tk, xk, inputs)

            llist.append(l.item())
            if self.break_diverge_early and l.item() > llist[0] * 100:
                print(f"Diverged with loss {l.item():.4f}, stopping...")
                break

            # Print out the intermediate results and the loss
            if self.show_recon_progress == True and i % self.print_every == 0:
                print("iteration: ", i, " loss: ", l)
                out_img = self.crop(xk).detach().cpu().numpy()

                if out_img.shape[-1] == 64:
                    fc_img = fc.pre_plot(fc.stack_rgb_opt(out_img))
                else:
                    fc_img = fc.stack_rgb_opt_30(out_img)

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

    def forward(self, input):
        input = torch.squeeze(input)
        assert len(input.shape) in {4, 3}, "Only accepts meas stack, or batch of stacks"

        if len(input.shape) == 4:
            output = torch.stack(
                [self.run(input[i, ...])[0] for i in range(input.shape[0])], 0
            )
        else:
            output = self.run(input)[0]

        return output