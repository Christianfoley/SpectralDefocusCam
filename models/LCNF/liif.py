"here, we use the LIIF method, modified from the LIIF algorithm: https://github.com/yinboc/liif"
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.LCNF.helpers as helpers
from models.LCNF.helpers import register

import utils.lcnf_utils as utils
from utils.lcnf_utils import make_coord


@register("liif")
class LIIF(nn.Module):
    def __init__(
        self,
        encoder_specs,
        imnet_spec=None,
        enhancements={},
        device=None,
    ):
        super().__init__()
        self.local_ensemble = enhancements.get("local_ensemble", True)
        self.feat_unfold = enhancements.get("feat_unfold", True)
        self.cell_decode = enhancements.get("cell_decode", True)
        self.encoord_dim = enhancements.get("encoord_dim", True)
        self.radialencode = enhancements.get("radialencode", True)
        self.angle_num = enhancements.get("angle_num", True)
        self.device = device

        self.defocus_encoders = []
        for spec in encoder_specs:
            self.defocus_encoders.append(helpers.make(spec))

        # self.DF_dim = self.encoder_DF.out_dim
        # self.DPC_dim = self.encoder_DPC.out_dim

        self.imnet = None
        if imnet_spec is not None:
            imnet_in_dim = sum([enc.out_dim for enc in self.defocus_encoders])

            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2  # attach coordinate
            if self.cell_decode:
                imnet_in_dim += 2

            self.imnet = helpers.make(imnet_spec, args={"in_dim": imnet_in_dim})

        # register list-hidden modules
        self.register_modules(self.defocus_encoders, "encoder")

    def to(self, device):
        # weird hacky thing i have to do since this model inits new tensors (BAD)
        self.device = device
        return self.cuda(device)

    def gen_feat(self, inp):  # generate latent code for the training
        self.feat = []
        for i, encoder in enumerate(self.defocus_encoders):
            self.feat.append(encoder(inp[:, i, :]))  # b, foc, c, y, x
        self.feat = torch.cat(self.feat, axis=1)
        return self.feat

    def query_latent(self, coord, cell=None):
        feat = self.feat  # generated latent code

        if self.imnet is None:
            ret = F.grid_sample(
                feat, coord.flip(-1).unsqueeze(1), mode="nearest", align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]
            )  # feature unfolding the neighbor 3*3 latent code

        if self.local_ensemble:  # used for the local ensemble
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1]), divide by 2 for radius
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # set coordinates for latent codes
        feat_coord = (
            make_coord(feat.shape[-2:], flatten=False)
            .cuda(self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        preds = (
            []
        )  # collect predictions for queried point based on different latent code
        areas = []  # collect areas as weights
        for vx in vx_lst:  # here, the [-1, 1] is used for local ensemble
            for vy in vy_lst:
                coord_ = coord.clone()  # this is coordinates of the queried points
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(  # pick the latent code related to the high-resolution images from the low-res latent space
                    feat,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[
                    :, :, 0, :
                ].permute(
                    0, 2, 1
                )
                q_coord = F.grid_sample(  # pick the corresponding coordinates
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)
                rel_coord = coord - q_coord  # calculate the relative coordinates
                rel_coord[:, :, 0] *= feat.shape[-2]  # times a scale factor
                rel_coord[:, :, 1] *= feat.shape[-1]

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat(
                        [q_feat, rel_cell], dim=-1
                    )  # here, it's the input feature + cell size
                else:
                    inp = [q_feat]

                inp = torch.cat(
                    [inp, rel_coord], dim=-1
                )  # here, it's the input feature + cell size + coordinates to MLP

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:  # diagonal area since larger area -> far from datapoint
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(
                -1
            )  # output with weighted sum
        return ret

    def forward(self, x: torch.Tensor, res_scaling=1, coord_batch_size=30000):
        # build coordinate sampling
        b, ims, c, h, w = x.shape
        h = x.shape[-2] * res_scaling
        w = x.shape[-1] * res_scaling
        coord = make_coord((h, w)).to(self.device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        # run prediction across coordinates
        coord, cell = coord.unsqueeze(0), cell.unsqueeze(0)
        n, ql = coord.shape[1], 0

        self.gen_feat(x)
        preds = []
        while ql < n:
            qr = min(ql + coord_batch_size, n)
            pred = self.query_latent(coord[:, ql:qr, :], cell[:, ql:qr, :])
            preds.append(pred)
            ql = qr
        preds = torch.cat(preds, dim=1)
        preds = preds.view(b, self.imnet.out_dim, h, w)  # b, c(lmbda), h, w
        return preds

    def pred_coord(self, inp, coord, cell):  # output of prediction
        self.gen_feat(inp)
        return self.query_latent(coord, cell)

    def register_modules(self, module_list, name):
        """
        Helper function that registers modules stored in a list to the model object so that the can
        be seen by PyTorch optimizer.

        Used to enable model graph creation with non-sequential model types and dynamic layer numbers

        :param list(torch.nn.module) module_list: list of modules to register/make visible
        :param str name: name of module type
        """
        for i, module in enumerate(module_list):
            self.add_module(f"{name}_{str(i)}", module)
