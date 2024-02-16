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
        w1 = C * (x[1::2, :, :] + x[0::2, :, :])
        w2 = soft_py(C * (x[1::2, :, :] - x[0::2, :, :]), thresh)
    elif ax == 1:
        w1 = C * (x[:, 1::2, :] + x[:, 0::2, :])
        w2 = soft_py(C * (x[:, 1::2, :] - x[:, 0::2, :]), thresh)
    elif ax == 2:
        w1 = C * (x[:, :, 1::2] + x[:, :, 0::2])
        w2 = soft_py(C * (x[:, :, 1::2] - x[:, :, 0::2]), thresh)
    return w1, w2


def iht3(w1, w2, ax, shift, shape, device):

    C = 1.0 / torch.sqrt(torch.tensor(2.0))
    y = torch.zeros(shape, device=device)

    x1 = C * (w1 - w2)
    x2 = C * (w1 + w2)
    if ax == 0:
        y[0::2, :, :] = x1
        y[1::2, :, :] = x2

    if ax == 1:
        y[:, 0::2, :] = x1
        y[:, 1::2, :] = x2
    if ax == 2:
        y[:, :, 0::2] = x1
        y[:, :, 1::2] = x2

    if shift == True:
        y = torch.roll(y, 1, dims=ax)
    return y


def tv3dApproxHaar(x, tau, alpha_w, alpha_x):
    D = 3
    fact = torch.sqrt(torch.tensor(2.0)).to(x.device) * 2

    thresh = D * tau * fact

    y = torch.zeros_like(x, device=x.device)
    for ax in range(0, len(x.shape)):
        if ax == 2:
            t_scale = alpha_w
        elif ax == 0:
            t_scale = alpha_x
        else:
            t_scale = 1

        w0, w1 = ht3(x, ax, False, thresh * t_scale)
        w2, w3 = ht3(x, ax, True, thresh * t_scale)

        t1 = iht3(w0, w1, ax, False, x.shape, x.device)
        t2 = iht3(w2, w3, ax, True, x.shape, x.device)
        y = y + t1 + t2

    y = y / (2 * D)
    return y
