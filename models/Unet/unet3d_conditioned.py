import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.diffuser_utils import *


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=False,
        norm="batch",
        residual=True,
        activation="selu",
        transpose=False,
    ):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose
        if self.dropout:
            self.dropout1 = nn.Dropout3d(p=0.05)
            self.dropout2 = nn.Dropout3d(p=0.05)
        self.norm1 = None
        self.norm2 = None
        if norm == "batch":
            self.norm1 = nn.BatchNorm3d(out_channels)
            self.norm2 = nn.BatchNorm3d(out_channels)
        elif norm == "instance":
            self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm == "mixed":
            self.norm1 = nn.BatchNorm3d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm == "local_response":
            self.norm1 = nn.LocalResponseNorm(size=5)
            self.norm2 = nn.BatchNorm3d(out_channels, affine=True)
        if self.transpose:
            self.conv1 = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=3, padding=1
            )
            self.conv2 = nn.ConvTranspose3d(
                out_channels, out_channels, kernel_size=3, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if self.activation == "relu":
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == "leakyrelu":
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == "elu":
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == "selu":
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()

    def forward(self, x):
        ox = x
        x = self.conv1(x)
        if self.dropout:
            x = self.dropout1(x)
        if self.norm1:
            x = self.norm1(x)
        x = self.actfun1(x)
        x = self.conv2(x)
        if self.dropout:
            x = self.dropout2(x)
        if self.norm2:
            x = self.norm2(x)
        if self.residual:
            x[:, 0 : min(ox.shape[1], x.shape[1]), :, :] += ox[
                :, 0 : min(ox.shape[1], x.shape[1]), :, :
            ]
        x = self.actfun2(x)
        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))
        return x


class Unet(nn.Module):
    def __init__(
        self,
        psfs,
        mask,
        n_channel_in=1,
        n_channel_out=1,
        norm="batch",
        residual=False,
        down="conv",
        up="tconv",
        activation="selu",
        condition_first_last_only=False,
    ):
        super(Unet, self).__init__()
        self.residual = residual
        if down == "maxpool":
            self.down1 = nn.MaxPool3d(kernel_size=2)
            self.down2 = nn.MaxPool3d(kernel_size=2)
            self.down3 = nn.MaxPool3d(kernel_size=2)
            self.down4 = nn.MaxPool3d(kernel_size=2)
        elif down == "avgpool":
            self.down1 = nn.AvgPool3d(kernel_size=2)
            self.down2 = nn.AvgPool3d(kernel_size=2)
            self.down3 = nn.AvgPool3d(kernel_size=2)
            self.down4 = nn.AvgPool3d(kernel_size=2)
        elif down == "conv":
            self.down1 = nn.Conv3d(32, 32, kernel_size=2, stride=2, groups=32)
            self.down2 = nn.Conv3d(64, 64, kernel_size=2, stride=2, groups=64)
            self.down3 = nn.Conv3d(128, 128, kernel_size=2, stride=2, groups=128)
            self.down4 = nn.Conv3d(256, 256, kernel_size=2, stride=2, groups=256)
            self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
            self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
            self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
            self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25
            self.down1.bias.data = 0.01 * self.down1.bias.data + 0
            self.down2.bias.data = 0.01 * self.down2.bias.data + 0
            self.down3.bias.data = 0.01 * self.down3.bias.data + 0
            self.down4.bias.data = 0.01 * self.down4.bias.data + 0
        if up == "bilinear" or up == "nearest":
            self.up1 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up2 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up3 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up4 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
        elif up == "tconv":
            self.up1 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2, groups=256)
            self.up2 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2, groups=128)
            self.up3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, groups=64)
            self.up4 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2, groups=32)
            self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
            self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
            self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
            self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25
            self.up1.bias.data = 0.01 * self.up1.bias.data + 0
            self.up2.bias.data = 0.01 * self.up2.bias.data + 0
            self.up3.bias.data = 0.01 * self.up3.bias.data + 0
            self.up4.bias.data = 0.01 * self.up4.bias.data + 0

        # ------------ Condition each layer on PSFS ------------#
        self.psfs = psfs.to(torch.float32)
        psf_conditions = []
        z_dims = [32, 16, 8, 4, 2]

        psfs = pad_yx_to_multiple(psfs.to(torch.float32), multiple=32)
        if condition_first_last_only:
            psfs = psfs[0 :: psfs.shape[0] - 1]
        self.depth = psfs.shape[0]

        for i in range(5):
            psf_conditions.append(
                psfs[None, :, None, ...].repeat(1, 1, z_dims[i], 1, 1)
            )
            psfs = F.avg_pool2d(psfs, kernel_size=2, stride=2)

        self.psfs1 = nn.parameter.Parameter(psf_conditions[0], False)
        self.psfs2 = nn.parameter.Parameter(psf_conditions[1], False)
        self.psfs3 = nn.parameter.Parameter(psf_conditions[2], False)
        self.psfs4 = nn.parameter.Parameter(psf_conditions[3], False)
        self.psfs5 = nn.parameter.Parameter(psf_conditions[4], False)
        del psfs, psf_conditions

        # ----------- Make mask  parameter ------------#
        self.mask = nn.parameter.Parameter(mask, False)
        self.DIMS0 = mask.shape[-2]  # Image Dimensions
        self.DIMS1 = mask.shape[-1]  # Image Dimensions
        self.PAD_SIZE0 = int((self.DIMS0))  # Pad size
        self.PAD_SIZE1 = int((self.DIMS1))  # Pad size

        # ------------------------------------------------------#

        self.conv1 = ConvBlock(
            n_channel_in + self.depth,
            32,
            norm=norm,
            residual=residual,
            activation=activation,
        )
        self.conv2 = ConvBlock(
            32 + self.depth, 64, norm=norm, residual=residual, activation=activation
        )
        self.conv3 = ConvBlock(
            64 + self.depth, 128, norm=norm, residual=residual, activation=activation
        )
        self.conv4 = ConvBlock(
            128 + self.depth, 256, norm=norm, residual=residual, activation=activation
        )
        self.conv5 = ConvBlock(
            256 + self.depth, 256, norm=norm, residual=residual, activation=activation
        )
        self.conv6 = ConvBlock(
            2 * 256, 128, norm=norm, residual=residual, activation=activation
        )
        self.conv7 = ConvBlock(
            2 * 128, 64, norm=norm, residual=residual, activation=activation
        )
        self.conv8 = ConvBlock(
            2 * 64, 32, norm=norm, residual=residual, activation=activation
        )
        self.conv9 = ConvBlock(
            2 * 32, n_channel_out, norm=norm, residual=residual, activation=activation
        )
        if self.residual:
            self.convres = ConvBlock(
                n_channel_in,
                n_channel_out,
                norm=norm,
                residual=residual,
                activation=activation,
            )

    def Hadj(self, b, h, mask):
        """
        LSI spatially invariant convolution adjoint method

        Parameters
        ----------
        b : torch.Tensor
            simulated (or not) measurement (b, n, 1, y, x)
        h : torch.Tensor
            measured or simulated psf data (n, y, x)
        mask : torch.Tensor
            spectral calibration matrix/mask (c, y, x)

        Returns
        -------
        torch.Tensor
            LSI system adjoint
        """
        Hconj = torch.conj(fft_psf(self, h))
        B_adj = fft_im(pad_zeros_torch(self, b * mask))
        v_hat = crop_forward(self, torch.fft.ifft2(Hconj * B_adj).real)
        return v_hat

    def fwd_pass(self, x):
        c0 = x

        c1 = self.conv1(
            torch.cat((x, self.psfs1.repeat(x.shape[0], 1, 1, 1, 1)), dim=1)
        )
        x = self.down1(c1)

        c2 = self.conv2(
            torch.cat((x, self.psfs2.repeat(x.shape[0], 1, 1, 1, 1)), dim=1)
        )
        x = self.down2(c2)

        c3 = self.conv3(
            torch.cat((x, self.psfs3.repeat(x.shape[0], 1, 1, 1, 1)), dim=1)
        )
        x = self.down3(c3)

        c4 = self.conv4(
            torch.cat((x, self.psfs4.repeat(x.shape[0], 1, 1, 1, 1)), dim=1)
        )
        x = self.down4(c4)

        x = self.conv5(torch.cat((x, self.psfs5.repeat(x.shape[0], 1, 1, 1, 1)), dim=1))

        x = self.up1(x)
        x = torch.cat([x, c4], 1)  # x[:,0:128]*x[:,128:256],
        x = self.conv6(x)
        x = self.up2(x)

        x = torch.cat([x, c3], 1)  # x[:,0:64]*x[:,64:128],
        x = self.conv7(x)
        x = self.up3(x)

        x = torch.cat([x, c2], 1)  # x[:,0:32]*x[:,32:64],
        x = self.conv8(x)
        x = self.up4(x)

        x = torch.cat([x, c1], 1)  # x[:,0:16]*x[:,16:32],
        x = self.conv9(x)

        if self.residual:
            x = torch.add(x, self.convres(c0))
        return x[:, 0, ...]  # remove image dimension

    def forward(self, x):
        # spectral projection & deconvolution
        x = self.Hadj(x, self.psfs, self.mask)

        # pass through conditioned unet
        x = F.pad(x, (0, 0, 0, 0, 1, 1), "constant", 0).to(torch.float32)
        x, padding = pad_yx_to_multiple(x, multiple=32, ret_padding=True)
        x = self.fwd_pass(x)
        x = x[
            ...,
            1:-1,
            padding[2] : x.shape[-2] - padding[3],
            padding[0] : x.shape[-1] - padding[1],
        ]

        return x
