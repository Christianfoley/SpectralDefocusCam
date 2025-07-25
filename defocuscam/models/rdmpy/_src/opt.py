"""
Implements all optimization functions. Primarily used by calibrate.py and deblur.py
"""
import pathlib
import gc

import numpy as np
import torch
import torch.fft as fft
import matplotlib.pyplot as plt

from tqdm import tqdm

from kornia.filters.sobel import spatial_gradient

from . import util, seidel
from .. import blur

dirname = str(pathlib.Path(__file__).parent.absolute())


def estimate_coeffs_blurstack(
    calib_image_stack,
    psf_list,
    sys_params,
    fit_params,
    show_psfs=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Estimate the Seidel coefficients of the optical system given a stack calibration images of
    randomly scattered PSFs and their locations. Stack is presumed to be taken at multiple blur
    levels of equal incremental amounts of defocus.


    Parameters
    ----------
    calib_image : np.ndarray
        Calibration image stack of randomly scattered PSFs (n, y, x).

    psf_list : list of tuples
        List of PSF locations in the calibration image. Expecting xy coordinates,
        not rowcol coordinates.

    sys_params : dict
        Dictionary of optical system parameters.

    fit_params : dict
        Dictionary of fitting parameters.

    show_psfs : bool, optional
        Whether to show the estimated PSFs. The default is False.

    verbose : bool, optional
        Whether to print out progress. The default is True.

    device : torch.device, optional
        Device to run the calibration on. The default is torch.device("cpu").

    Returns
    -------
    final_coeffs : torch.Tensor
        Seidel coefficients of the optical system.

    """

    psfs_gt_stack = torch.tensor(calib_image_stack, device=device).float()
    stack_depth = psfs_gt_stack.shape[0]

    n_seidel = 6
    anchor_blur = fit_params.get("enforce_anchor_blur", 1)
    if "enforce_anchor_blur" in fit_params:
        n_seidel = 5

    if fit_params["seidel_init"] is not None:
        coeffs = torch.tensor(fit_params["seidel_init"], device=device)
    else:
        if fit_params["init"] == "zeros":
            coeffs = torch.zeros((n_seidel, 1), device=device)
        elif fit_params["init"] == "random":
            coeffs = torch.rand((n_seidel, 1), device=device)
        else:
            raise NotImplementedError
    coeffs.requires_grad = True

    optimizer = torch.optim.Adam([coeffs], lr=fit_params["lr"])
    l2_loss_fn = torch.nn.MSELoss()
    l1_loss_fn = torch.nn.L1Loss()

    anchor_blur_level = fit_params.get("anchor_blur_idx", stack_depth // 2)

    if fit_params["plot_loss"]:
        losses = []

    iterations = (
        tqdm(range(fit_params["iters"])) if verbose else range(fit_params["iters"])
    )

    for iter in iterations:
        # estimate psfs over stack
        psfs_estimate_stack = []
        for blur_idx in range(stack_depth):
            blur_scale = torch.ones(6, 1, device=device)
            blur_scale[5, 0] = blur_idx / anchor_blur_level

            if n_seidel == 6:
                idx_coeffs = coeffs
            else:
                idx_coeffs = torch.cat(
                    (coeffs, torch.ones(1, 1, device=device) * anchor_blur)
                )
            psfs_estimate_stack.append(
                sum(
                    seidel.compute_psfs(
                        idx_coeffs * blur_scale,
                        desired_list=psf_list[anchor_blur_level],
                        stack=False,
                        sys_params=sys_params,
                        device=coeffs.device,
                    )
                ).float()
            )
        psfs_estimate_stack = torch.stack(psfs_estimate_stack, 0)

        # loss
        loss = l2_loss_fn(
            util.normalize(psfs_estimate_stack),
            util.normalize(psfs_gt_stack),
        ) + fit_params["reg"] * l1_loss_fn(coeffs, -coeffs)

        if fit_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if show_psfs:
        for i in range(stack_depth):
            psfs_est = psfs_estimate_stack[i]
            psfs_gt = psfs_gt_stack[i]

            plt.subplot(1, 2, 2)
            plt.tight_layout()
            plt.axis("off")
            plt.imshow(psfs_est.detach().cpu(), cmap="inferno")
            plt.gca().set_title("Seidel PSFs")

            plt.subplot(1, 2, 1)
            plt.tight_layout()
            plt.axis("off")
            plt.imshow(psfs_gt.detach().cpu(), cmap="inferno")
            plt.gca().set_title("Measured PSFs")
            plt.show()

    if fit_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

    # memory cleanup
    del psfs_estimate_stack
    del psfs_gt_stack

    if n_seidel != 6:
        coeffs = torch.cat((coeffs, torch.ones(1, 1, device=device) * anchor_blur))

    final_coeffs = []
    for blur_idx in range(stack_depth):
        blur_scale = torch.ones(6, 1, device=device)
        blur_scale[5, 0] = blur_idx / anchor_blur_level
        final_coeffs.append((coeffs * blur_scale).detach())

    return torch.stack(final_coeffs, 0)


def estimate_coeffs(
    calib_image,
    psf_list,
    sys_params,
    fit_params,
    show_psfs=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Estimate the Seidel coefficients of the optical system given a calibration image of
    randomly scattered PSFs and their locations.


    Parameters
    ----------
    calib_image : np.ndarray
        Calibration image of randomly scattered PSFs.

    psf_list : list of tuples
        List of PSF locations in the calibration image. Expecting xy coordinates,
        not rowcol coordinates.

    sys_params : dict
        Dictionary of optical system parameters.

    fit_params : dict
        Dictionary of fitting parameters.

    show_psfs : bool, optional
        Whether to show the estimated PSFs. The default is False.

    verbose : bool, optional
        Whether to print out progress. The default is True.

    device : torch.device, optional
        Device to run the calibration on. The default is torch.device("cpu").

    Returns
    -------
    final_coeffs : torch.Tensor (or list if fit_params['get_inter_seidels'] is True)
        Seidel coefficients of the optical system.

    """

    psfs_gt = torch.tensor(calib_image, device=device).float()
    if fit_params["seidel_init"] is not None:
        coeffs = torch.tensor(fit_params["seidel_init"], device=device)
    else:
        if fit_params["init"] == "zeros":
            coeffs = torch.zeros((fit_params["num_seidel"], 1), device=device)
        elif fit_params["init"] == "random":
            coeffs = torch.rand((fit_params["num_seidel"], 1), device=device)
        else:
            raise NotImplementedError
    coeffs.requires_grad = True

    optimizer = torch.optim.Adam([coeffs], lr=fit_params["lr"])
    l2_loss_fn = torch.nn.MSELoss()
    l1_loss_fn = torch.nn.L1Loss()

    blur_constraint = fit_params.get("enforce_blur", 0)

    if fit_params["plot_loss"]:
        losses = []

    if fit_params["get_inter_seidels"]:
        inter_seidels = []

    iterations = (
        tqdm(range(fit_params["iters"])) if verbose else range(fit_params["iters"])
    )

    for iter in iterations:
        # forward pass
        if coeffs.shape[0] < 6:
            psfs_estimate = seidel.compute_psfs(
                torch.cat(
                    (
                        coeffs,
                        torch.ones(6 - coeffs.shape[0], 1, device=device)
                        * blur_constraint,
                    )
                ),
                desired_list=psf_list,
                stack=False,
                sys_params=sys_params,
                device=coeffs.device,
            )
        else:
            psfs_estimate = seidel.compute_psfs(
                coeffs,
                desired_list=psf_list,
                stack=False,
                sys_params=sys_params,
                device=coeffs.device,
            )
        # loss
        loss = l2_loss_fn(
            util.normalize(sum(psfs_estimate).float()), util.normalize(psfs_gt)
        ) + fit_params["reg"] * l1_loss_fn(coeffs, -coeffs)

        if fit_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if fit_params["get_inter_seidels"]:
            if coeffs.shape[0] < 6:
                inter_seidels += [
                    torch.cat(
                        (
                            coeffs,
                            torch.ones(6 - coeffs.shape[0], 1, device=device)
                            * blur_constraint,
                        )
                    ).detach()
                ]
            else:
                inter_seidels += [coeffs.detach().cpu().detach()]

    if show_psfs:
        psf_est = sum(psfs_estimate) / (len(psfs_estimate))

        plt.subplot(1, 2, 2)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(psf_est.detach().cpu(), cmap="inferno")
        plt.gca().set_title("Seidel PSFs")

        plt.subplot(1, 2, 1)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(psfs_gt.detach().cpu(), cmap="inferno")
        plt.gca().set_title("Measured PSFs")
        plt.show()

    if fit_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()
    del psfs_estimate
    del psfs_gt

    if fit_params["get_inter_seidels"]:
        final_coeffs = inter_seidels
    else:
        if coeffs.shape[0] < 6:
            final_coeffs = torch.cat(
                (
                    coeffs,
                    torch.ones(6 - coeffs.shape[0], 1, device=device) * blur_constraint,
                )
            ).detach()
        else:
            final_coeffs = coeffs.detach()

    return final_coeffs


def image_recon(
    measurement,
    psf_data,
    model,
    opt_params,
    warm_start=None,
    use_batch_conv=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Deblurs an image using either deconvolution or ring deconvolution

    Parameters
    ----------
    measurement : torch.Tensor
        Measurement to be deblurred. Should be (N,N).

    psf_data : torch.Tensor
        Stack of rotationatal Fourier transforms of the PSFs if model is 'lri',
        otherwise single center PSF if model is 'lsi'.

    model : str
        Either 'lri' or 'lsi'.

    opt_params : dict
        Dictionary of optimization parameters.

    warm_start : torch.Tensor, optional
        Warm start for the optimization. The default is None.

    use_batch_conv : bool, optional
        Whether to use batched lri convolution. The default is False.

    verbose : bool, optional
        Whether to print out progress. The default is True.

    device : torch.device, optional
        Device to run the reconstruction on. The default is torch.device("cpu").

    Returns
    -------
    estimate : torch.Tensor
        Deblurred image. Will be (N,N).

    """
    dim = measurement.shape

    if warm_start is not None:
        estimate = warm_start.clone()
    else:
        if opt_params["init"] == "measurement":
            estimate = measurement.clone()
        elif opt_params["init"] == "zero":
            estimate = torch.zeros(dim, device=device)
        elif opt_params["init"] == "noise":
            estimate = torch.randn(dim, device=device)
        else:
            raise NotImplementedError

    estimate.requires_grad = True

    if opt_params["optimizer"] == "adam":
        optimizer = torch.optim.Adam([estimate], lr=opt_params["lr"])
    elif opt_params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD([estimate], lr=opt_params["lr"])
    else:
        raise NotImplementedError

    loss_fn = torch.nn.MSELoss()
    crop = opt_params["crop"]

    losses = []

    if opt_params["plot_loss"]:
        losses = []

    iterations = (
        tqdm(range(opt_params["iters"])) if verbose else range(opt_params["iters"])
    )

    crop = lambda x: x[..., crop:-crop, crop:-crop] if opt_params["crop"] > 0 else x

    for it in iterations:
        # forward pass and loss
        if model == "lsi":
            measurement_guess = blur.convolve(estimate, psf_data)
        elif use_batch_conv:
            measurement_guess = blur.batch_ring_convolve(
                estimate[None, None, ...], psf_data, device=device
            )[0, 0]
        else:
            measurement_guess = blur.ring_convolve(estimate, psf_data, device=device)

        loss = (
            loss_fn(crop(measurement_guess), crop(measurement))
            + tv(estimate, opt_params["tv_reg"])
            + opt_params["l2_reg"] * torch.norm(estimate)
        )

        if opt_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # project onto [0,1]
        estimate.data[estimate.data < 0] = 0
        estimate.data[estimate.data > 1] = 1

    if opt_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

    final = util.normalize(estimate.detach().cpu().float().numpy().copy())

    del estimate
    gc.collect()
    torch.cuda.empty_cache()

    return final


def video_recon(
    measurement_stack,
    psf_data,
    model,
    opt_params,
    batch_size=4,
    prev_fram_warm_start=True,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Reconstructs a video frame by frame.

    Parameters
    ----------
    measurement_stack : torch.Tensor
        Stack of measurements. Should be (M,N,N) where M is the number of frames.

    psf_data : torch.Tensor
        Stack of rotationatal Fourier transforms of the PSFs if model is 'lri',
        otherwise single center PSF if model is 'lsi'.

    model : str
        Either 'lri' or 'lsi'.

    opt_params : dict
        Dictionary of optimization parameters.

    batch_size : int
        size for stack batching

    prev_fram_warm_start : bool, optional
        Whether to use the previous frame as a warm start. The default is True.

    verbose : bool, optional
        Whether to print out progress. The default is True.

    device : torch.device
        Device to run the reconstruction on.

    Returns
    -------
    measurement_stack : torch.Tensor
        Reconstructed video. Will be (M,N,N)

    """
    start_idx = 0
    if prev_fram_warm_start:
        # run the first frame to initialize warm start
        measurement_stack[0] = image_recon(
            measurement_stack[0, :, :],
            psf_data,
            model,
            opt_params=opt_params,
            verbose=verbose,
            device=device,
        )
        start_idx = 1

    # process stack chunk-wise
    for i in range(start_idx, measurement_stack.shape[0], batch_size):
        if prev_fram_warm_start:
            measurement_stack[i : i + batch_size, :, :] = image_recon(
                measurement_stack[i : i + batch_size, :, :],
                psf_data,
                model,
                opt_params=opt_params,
                warm_start=measurement_stack[i - 1 : batch_size - 1, :, :],
                device=device,
            )
        else:
            measurement_stack[i : i + batch_size, :, :] = image_recon(
                measurement_stack[i : i + batch_size, :, :],
                psf_data,
                model,
                opt_params=opt_params,
                verbose=verbose,
                device=device,
            )

    return measurement_stack


def blind_recon(
    measurement, opt_params, sys_params, verbose=True, device=torch.device("cpu")
):
    """
    Deblurs an image using a blind deconvolution approach (no calibration needed).

    Parameters
    ----------
    measurement : torch.Tensor
        Measurement to be deblurred. Should be (M,N).

    opt_params : dict
        Dictionary of optimization parameters.

    sys_params : dict
        Dictionary of optical system parameters.

    verbose : bool, optional
        Whether to print out progress. The default is True.

    device : torch.device, optional
        Device to run the reconstruction on. The default is torch.device("cpu").

    Returns
    -------
    recon : torch.Tensor
        Deblurred image. Will be (M,N).

    psfs_estimate : np.ndarray
        Estimated PSF. Will be (M,N).

    """

    if opt_params["seidel_init"] is None:
        coeffs = torch.zeros((1, 1), device=device)

    coeffs.requires_grad = True

    optimizer = torch.optim.Adam([coeffs], lr=opt_params["lr"])

    if opt_params["plot_loss"]:
        losses = []

    iterations = (
        tqdm(range(opt_params["iters"])) if verbose else range(opt_params["iters"])
    )

    for iter in iterations:
        # forward pass
        psfs_estimate = seidel.compute_psfs(
            torch.cat((coeffs, torch.zeros(5, 1, device=device))),
            desired_list=[(0, 0)],
            sys_params=sys_params,
            device=coeffs.device,
        )[0]
        recon = wiener(
            ((measurement - 0.5) * 2), psfs_estimate, balance=opt_params["balance"]
        )
        recon = (recon / 2) + 0.5  # back-scale

        # loss (maximizing acutance)
        loss = -torch.mean(
            torch.abs(spatial_gradient(recon[None, None, :, :], mode="diff"))
        )

        # pdb.set_trace()
        if opt_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if opt_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

    if verbose:
        print("Fitted seidel coefficient: " + str(coeffs[0].detach().cpu()))

    return (
        recon.detach().cpu().numpy(),
        psfs_estimate.detach().cpu().numpy(),
    )


def tv(img, weight):
    tv_h = ((img[1:, :] - img[:-1, :]).abs()).sum()
    tv_w = ((img[:, 1:] - img[:, :-1]).abs()).sum()
    return weight * (tv_h + tv_w)


def center_crop(measurement, des_shape):
    # Center crop
    m_center = (measurement.shape[0] // 2, measurement.shape[1] // 2)
    left, right, up, down = (
        m_center[1] - des_shape[1] // 2,
        m_center[1] + int(np.round(des_shape[1] / 2)),
        m_center[0] - des_shape[0] // 2,
        m_center[0] + int(np.round(des_shape[0] / 2)),
    )
    # TODO: Debug this for images of an odd size.
    measurement = measurement[left:right, up:down]
    return measurement


def wiener(image, psf, balance=5e-4, reg=None, is_real=True, clip=True):
    """
    Applies a wiener filter to an image using the psf.

    Parameters
    ----------
    image : torch.Tensor
        Image to be deblurred. Should be (M,N).

    psf : torch.Tensor
        PSF to be used for deblurring. Should be (M,N).

    balance : float, optional
        Balance parameter for the wiener filter. The default is 5e-4.

    reg : torch.Tensor, optional
        Regularization parameter for the wiener filter. The default is None.

    is_real : bool, optional
        Whether the PSF is real or complex. The default is True.

    clip : bool, optional
        Whether to clip the output to [-1,1]. The default is True.

    Returns
    -------
    deconv : torch.Tensor
        Deblurred image. Will be (M,N).

    Notes
    -----
    This function is adapted from the `wiener` function in the `restoration` module of
    scikit-image. https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/restoration/deconvolution.py


    """

    if reg is None:
        reg, _ = laplacian(image.ndim, image.shape, is_real=is_real)
        reg = reg.to(device=image.device)

    trans_func = ir2tf(psf, image.shape, is_real=is_real).to(device=image.device)

    wiener_filter = torch.conj(trans_func) / (
        torch.abs(trans_func) ** 2 + balance * torch.abs(reg) ** 2
    )

    deconv = fft.irfftn(wiener_filter * fft.rfftn(image))

    if clip:
        deconv[deconv > 1] = 1
        deconv[deconv < -1] = -1

    return deconv


def laplacian(ndim, shape, is_real=True):
    """
    Computes the Laplacian operator in the Fourier domain.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the image.

    shape : tuple
        Shape of the image.

    is_real : bool, optional
        Whether the PSF is real or complex. The default is True.

    Returns
    -------
    laplacian : torch.Tensor
        Laplacian operator in the Fourier domain.

    impr : torch.Tensor
        Laplacian operator in the spatial domain.

    Notes
    -----
    This function is taken from the `laplacian` function in the `restoration` module of
    scikit-image. https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/restoration/deconvolution.py
    """

    impr = torch.zeros([3] * ndim)
    for dim in range(ndim):
        idx = tuple(
            [slice(1, 2)] * dim + [slice(None)] + [slice(1, 2)] * (ndim - dim - 1)
        )
        impr[idx] = torch.tensor([-1.0, 0.0, -1.0]).reshape(
            [-1 if i == dim else 1 for i in range(ndim)]
        )
    impr[(slice(1, 2),) * ndim] = 2.0 * ndim
    return ir2tf(impr, shape, is_real=is_real), impr


def ir2tf(imp_resp, shape, dim=None, is_real=True):
    """
    Implements the ir2tf function from the `restoration` module of scikit-image.

    Parameters
    ----------
    imp_resp : torch.Tensor
        Impulse response of the system.

    shape : tuple
        Shape of the image.

    dim : int, optional
        Last axis to compute the FFT over. The default is None.

    is_real : bool, optional
        Whether the PSF is real or complex. The default is True.

    Returns
    -------
    tf : torch.Tensor
        Transfer function of the system.

    Notes
    -----
    This function is taken from the `ir2tf` function in the `restoration` module of
    scikit-image. https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/restoration/uft.py

    """

    if not dim:
        dim = imp_resp.ndim
    # Zero padding and fill
    if imp_resp.shape != shape:
        irpadded = torch.zeros(shape)
        irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    else:
        irpadded = imp_resp
    # Roll for zero convention of the fft to avoid the phase
    # problem. Work with odd and even size.
    for axis, axis_size in enumerate(imp_resp.shape):
        if axis >= imp_resp.ndim - dim:
            irpadded = torch.roll(
                irpadded, shifts=-int(np.floor(axis_size / 2)), dims=axis
            )

    return fft.rfftn(irpadded, dim=list(range(-dim, 0)))


class StackwiseMSELossTV(torch.nn.Module):
    """
    MSE loss with tv regularization applied stack-wise to a BNN or BCNN batch
    """

    def __init__(self, tv_reg, l2_reg, crop):
        """
        Build object for stackwise MSE loss with TV

        Parameters
        ----------
        tv_reg : float
            scaling parameter for TV regularization
        l2_reg : float
            scaling parameter for Ridge
        crop : int
            crop (on all sides of images) to apply
        """
        super().__init__()
        self.tv_reg = tv_reg
        self.l2_reg = l2_reg
        self.crop = crop

    def forward(self, pred, target, estimate):
        """
        Forward application of loss.

        Parameters
        ----------
        pred : torch.Tensor
            prediction tensor
        target : torch.Tensor
            Target/label tensor
        estimate : torch.Tensor
            Current estimate tensor (for regularization)

        Returns
        -------
        torch.Tensor
            Loss tensor, batchwise
        """
        assert (
            pred.size() == target.size()
        ), "'pred' and 'target' must have the same size."

        loss = torch.zeros((pred.shape[0],) + (1,) * (len(pred.shape) - 1))

        if self.crop is not 0:
            pred = pred[..., self.crop : -self.crop, self.crop : -self.crop]
            target = target[..., self.crop : -self.crop, self.crop : -self.crop]
            estimate = estimate[..., self.crop : -self.crop, self.crop : -self.crop]

        for i in range(loss.shape[0]):
            loss[i] = (
                torch.mean((pred[i, ...] - target[i, ...]) ** 2)
                + tv(estimate[i, ...], self.tv_reg)
                + self.l2_reg * torch.norm(estimate[i, ...])
            )

        return loss
