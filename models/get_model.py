import sys

sys.path.append("..")

from utils.diffuser_utils import *

import models.ensemble as ensemble
import models.forward as fm

import models.Unet.unet as Unet2d
import models.Unet.unet3d as Unet3d
import models.Unet.R2attunet as R2attunet3d
import models.LCNF.liif as liif
import models.fista.fista_spectral_cupy_batch as fista


def get_model(config, device, fwd_only=False, force_psfs=None):
    """
    Constructs a model from the given forward and recon model params. If data_precomputed
    is true, forward model will be "passthrough".

    Parameters
    ----------
    config : dict
        config dictionary with model hyperparams
    device : torch.Device
        device to place models on
    fwd_only : bool
        whether to return only an initialized forward model
    force_psfs : torch.tensor, optional
        psfs tensors to condition recon instead of the saved psfs, by default None

    Returns
    -------
    torch.nn.Module
        "ensemble" wrapper model for forward and recon models
    """
    fm_params = config["forward_model_params"]
    rm_params = config["recon_model_params"]
    rm_params["num_measurements"] = fm_params["stack_depth"]
    rm_params["blur_stride"] = fm_params["psf"]["stride"]

    # forward model
    mask = load_mask(
        path=config["mask_dir"],
        patch_crop_center=config["image_center"],
        patch_crop_size=config["patch_crop"],
        patch_size=config["patch_size"],
    )
    forward_model = fm.ForwardModel(
        mask,
        params=fm_params,
        psf_dir=config["psf_dir"],
        passthrough=config["data_precomputed"],
        device=device,
    )
    forward_model.init_psfs()

    # recon model
    if rm_params["model_name"] == "fista":
        if force_psfs is None:
            recon_model = fista.fista_spectral_numpy(
                forward_model.psfs,
                torch.tensor(mask),
                params=rm_params,
                device=device,
            )
        else:
            recon_model = fista.fista_spectral_numpy(
                force_psfs, torch.tensor(mask), params=rm_params, device=device
            )
    elif rm_params["model_name"] == "unet":
        recon_model = Unet3d.Unet(n_channel_in=rm_params["num_measurements"])
    elif rm_params["model_name"] == "unet2d":
        recon_model = Unet2d.Unet(
            n_channel_in=rm_params["num_measurements"],
            n_channel_out=rm_params["spectral_depth"],
        )
    elif rm_params["model_name"] == "r2attunet":
        recon_model = R2attunet3d.R2AttUnet(
            in_ch=rm_params["num_measurements"],
            t=rm_params.get("recurrence_t", 2),
        )
    elif rm_params["model_name"] == "lcnf":
        encoder_specs = [rm_params["encoder_specs"]] * rm_params["num_measurements"]
        recon_model = liif.LIIF(
            encoder_specs,
            rm_params["imnet_spec"],
            rm_params["enhancements"],
        )

    # build ensemble and load any pretrained weights
    full_model = ensemble.MyEnsemble(forward_model, recon_model)

    if config.get("preload_weights", False):
        full_model.load_state_dict(
            torch.load(config["checkpoint_dir"], map_location="cpu")
        )

    if fwd_only:
        return full_model.model1.to(device)
    return full_model.to(device)
