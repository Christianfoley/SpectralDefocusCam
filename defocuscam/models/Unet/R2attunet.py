# Credit for the below code goes to https://github.com/LeeJunHyun/Image_Segmentation
# Code has been altered to support 3d inputs
import torch.nn as nn
import torch


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t), Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class R2AttUnet(nn.Module):
    """
    Implementation of an attention augmented R2-unet. Changed to support 3d inputs.

    Structure shamelessly lifted from:
        https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, psfs, spec_chans=32, in_ch=1, output_ch=1, t=2):
        super(R2AttUnet, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        # ------------ Condition each layer on PSFS ------------#
        self.depth = psfs.shape[0]
        psfs = psfs[None, :, None, ...].repeat(1, 1, spec_chans, 1, 1).to(torch.float32)
        psf_conditions = [psfs]

        for i in range(3):
            psfs = self.Maxpool(psfs)
            psf_conditions.append(psfs)
        self.psfs1 = nn.parameter.Parameter(psf_conditions[0])
        self.psfs2 = nn.parameter.Parameter(psf_conditions[1])
        self.psfs3 = nn.parameter.Parameter(psf_conditions[2])
        self.psfs4 = nn.parameter.Parameter(psf_conditions[3])
        del psfs, psf_conditions
        # ------------------------------------------------------#

        self.RRCNN1 = RRCNN_block(ch_in=in_ch + self.depth, ch_out=32, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32 + self.depth, ch_out=64, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=64 + self.depth, ch_out=128, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=128 + self.depth, ch_out=256, t=t)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Att4 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Att3 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Att2 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32, t=t)

        self.Conv_1x1 = nn.Conv3d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x = torch.cat((x, self.psfs1), dim=1)

        x1 = self.RRCNN1(x)
        x2 = self.Maxpool(x1)

        x2 = torch.cat((x2, self.psfs2), dim=1)
        x2 = self.RRCNN2(x2)
        x3 = self.Maxpool(x2)

        x3 = torch.cat((x3, self.psfs3), dim=1)
        x3 = self.RRCNN3(x3)
        x4 = self.Maxpool(x3)

        x4 = torch.cat((x4, self.psfs4), dim=1)
        x4 = self.RRCNN4(x4)

        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(x3)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1[:, 0, ...]  # remove image dimension
