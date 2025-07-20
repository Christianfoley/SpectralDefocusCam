import torch


def soft_py(x, tau):
    threshed = torch.maximum(
        torch.abs(x) - tau, torch.tensor(0, dtype=x.dtype, device=x.device)
    )
    threshed = threshed * torch.sign(x)
    return threshed


def ht3(x, ax, shift, thresh):
    C = (1.0 / torch.sqrt(torch.tensor(2.0))).to(x.device)

    if shift == True:
        x = torch.roll(x, -1, dims=ax)
    if ax == 0:
        w1 = C * (x[..., 1::2, :, :] + x[..., 0::2, :, :])
        w2 = soft_py(C * (x[..., 1::2, :, :] - x[..., 0::2, :, :]), thresh)
    elif ax == 1:
        w1 = C * (x[..., :, 1::2, :] + x[..., :, 0::2, :])
        w2 = soft_py(C * (x[..., :, 1::2, :] - x[..., :, 0::2, :]), thresh)
    elif ax == 2:
        w1 = C * (x[..., :, :, 1::2] + x[..., :, :, 0::2])
        w2 = soft_py(C * (x[..., :, :, 1::2] - x[..., :, :, 0::2]), thresh)
    return w1, w2


def iht3(w1, w2, ax, shift, shape, device):
    C = 1.0 / torch.sqrt(torch.tensor(2.0))
    y = torch.zeros(shape, device=device)

    x1 = C * (w1 - w2)
    x2 = C * (w1 + w2)
    if ax == 0:
        y[..., 0::2, :, :] = x1
        y[..., 1::2, :, :] = x2
    if ax == 1:
        y[..., :, 0::2, :] = x1
        y[..., :, 1::2, :] = x2
    if ax == 2:
        y[..., :, :, 0::2] = x1
        y[..., :, :, 1::2] = x2

    if shift == True:
        y = torch.roll(y, 1, dims=ax)
    return y


def tv3dApproxHaar(x, tau, alpha_w, alpha_x, order="xyw"):
    """
    Computes a 3d tv soft thresholding using haar decompositions
    along the last 3 dimensions of tensor x

    Parameters
    ----------
    x : torch.Tensor
        tensor to apply TV thresholding to (..., x, y, c)
    tau : float
        threshold variable
    alpha_w : float
        thresholding weight along lambda dimension
    alpha_x : float
        thresholding weight along x dimension

    Returns
    -------
    torch.Tensor
        soft-thresholded 3d tensor
    """
    D = 3
    fact = torch.sqrt(torch.tensor(2.0)).to(x.device) * 2

    thresh = D * tau * fact

    y = torch.zeros_like(x, device=x.device)
    for ax in range(0, len(order)):
        # Determine threshold scaling by dimension
        if order[ax] == "x":
            t_scale = alpha_x
        elif order[ax] == "w":
            t_scale = alpha_w
        else:
            t_scale = 1

        # apply haar transform with scaled threshold
        w0, w1 = ht3(x, ax, False, thresh * t_scale)
        w2, w3 = ht3(x, ax, True, thresh * t_scale)

        # apply inverse haar transform
        t1 = iht3(w0, w1, ax, False, x.shape, x.device)
        t2 = iht3(w2, w3, ax, True, x.shape, x.device)
        y = y + t1 + t2

    y = y / (2 * D)
    return y
