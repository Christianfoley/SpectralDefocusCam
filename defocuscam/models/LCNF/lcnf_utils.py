import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter


class Averager:
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    elif t >= 60:
        return "{:.1f}m".format(t / 60)
    else:
        return "{:.1f}s".format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename="log.txt"):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), "a") as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip("/"))
    if os.path.exists(path):
        if remove and (
            basename.startswith("_")
            or input("{} exists, remove? (y/[n]): ".format(path)) == "y"
        ):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)  # make savefolder
    set_log_path(save_path)
    writer = SummaryWriter(
        os.path.join(save_path, "tensorboard")
    )  # make filename for the tensorboard
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return "{:.1f}M".format(tot / 1e6)
        else:
            return "{:.1f}K".format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {"sgd": SGD, "adam": Adam}[optimizer_spec["name"]]
    optimizer = Optimizer(param_list, **optimizer_spec["args"])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec["sd"])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)  # sampling distance
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """Convert the image to coord-RGB pairs.
    img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def to_pixel_samples2(img):
    """Convert the image to coord-value pairs
    img: Tensor, (5, H, W)"""
    coord = make_coord(img.shape[-2:])
    value = img.view(1, -1).permute(1, 0)  # here, change from the RGB color
    return coord, value


def encode_coordnate(delta_X, encoord_dim):
    "encode the coordinate using the cos and sin functions"
    dx1 = delta_X[:, :, 0]
    dx2 = delta_X[:, :, 1]
    coord_shape = delta_X.shape
    encode_dx = torch.zeros(*coord_shape[:-1], encoord_dim * 4)

    for i in range(encoord_dim):
        encode_dx[:, :, i * 4] = torch.sin(2 * math.exp(i + 1) * dx1)
        encode_dx[:, :, i * 4 + 1] = torch.cos(2 * math.exp(i + 1) * dx1)
        encode_dx[:, :, i * 4 + 2] = torch.sin(2 * math.exp(i + 1) * dx2)
        encode_dx[:, :, i * 4 + 3] = torch.cos(2 * math.exp(i + 1) * dx2)
    return encode_dx


def radial_encode(delta_X, encoord_dim, angle_num):
    "encode the coordinate using the radial encoding and also the cos and sin functions"
    # encode with radial encoding method
    delta_angle = int(360 / angle_num)
    angle_list = torch.arange(0, 360, delta_angle)

    dx1 = delta_X[:, :, 0]
    dx2 = delta_X[:, :, 1]

    rotate_angle = torch.ones([*dx1.shape, angle_num]) * angle_list
    rotate_angle = rotate_angle.cuda()
    rotated_coord = torch.zeros([*dx1.shape, angle_num * 2])
    rotated_coord = rotated_coord.cuda()
    # calculate the rotated coordinates
    for i in range(angle_num):
        rotated_coord[:, :, i * 2] = (
            torch.cos(rotate_angle[:, :, i]) * dx1
            - torch.sin(rotate_angle[:, :, i]) * dx2
        )
        rotated_coord[:, :, i * 2 + 1] = (
            torch.sin(rotate_angle[:, :, i]) * dx1
            + torch.cos(rotate_angle[:, :, i]) * dx2
        )

    # encode the rotated coordinates with sin(), cos() functions
    coord = torch.zeros(
        [*rotated_coord.shape[:-1], 2 * encoord_dim * (2 * rotate_angle.shape[-1])]
    )
    for i in range(angle_num):
        encode_dx = encode_coordnate(
            rotated_coord[:, :, i * 2 : (i + 1) * 2], encoord_dim
        )
        coord[:, :, i * 4 * encoord_dim : (i + 1) * 4 * encoord_dim] = encode_dx
    return coord


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == "benchmark":
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == "div2k":
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_latent(coord[:, ql:qr, :], cell[:, ql:qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred
