import sys, os, glob
import matplotlib.pyplot as plt
import numpy as np
import torch

import models.rdmpy.blur as blur
import models.rdmpy.calibrate as calibrate

import models.forward as forward
import data_utils.dataset as ds
import data_utils.preprocess_data as prep_data
import utils.psf_calibration_utils as psf_utils
import utils.diffuser_utils as diffuser_utils

device = "cuda:0"

psf_path = "../defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/10_26/multipos/alignment_psfs_telecent25um_10_26"

focus_levels = 3  # levels of focus in your measurements
estimate_method = (
    "median"  # mean if your points seem to shift very consistently, median if they dont
)
coord_method = "conv"  # method for estimating psf coordinates. Note "conv" may outperform but is slower
crop = (
    400,
    0,
    1500,
    1000,
)  # crop down to the general psf location to reduce computation
anchor_idx = 1  # index of first focus to draw vector from
delta_idx = 2  # index of second focus to draw vector through
conv_kern_sizes = [
    7,
    15,
    21,
]  # approximate psf sizes, used as kernel size in convolution

# Note: a "good" estimate is one with a clustered distribution
alignment_estimate = psf_utils.estimate_alignment_center(
    psf_path,
    focus_levels,
    estimate_method=estimate_method,
    coord_method=coord_method,
    anchor_foc_idx=anchor_idx,
    vector_foc_idx=delta_idx,
    crop=crop,
    verbose=True,
    plot=False,
)

dim = 768  # cropsize of calibration image

num_seidel = 5  # number of seidel coefficients to fit: 5 to disclude defocus, 6 to include defocus
calibrate_focus_level = 2  # focus level to run calibration on
fit_params = {
    "threshold": 0.4,  # proportion of max val for psf localization, lower = more sensitive
    "sys_center": alignment_estimate,  # alignment axis
    "enforce_blur": 2.5,
}


raw_datapath = "/home/cfoley_waller/10tb_extension/defocam/defocuscamdata/sample_data/"
processed_datapath = (
    "/home/cfoley_waller/defocam/defocuscamdata/sample_data_preprocessed_lri_11_2"
)

patchsize = (
    256,
    256,
)  # size of patches to chunk data - factor of cropsize for best results
num_channels = 30  # channels are linearly interpolated to fit this value from raw data - note same issue as above


mask = diffuser_utils.load_mask(
    path="/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/calib_matrix_full_10_18_2023_NEW_exp25.64ms_gain14.9_navg16_378-890/calibration_matrix.mat",
    patch_crop_center=[1050, 2023],  # center of mask and image
    patch_crop_size=[dim, dim],  # size of mask crop before downsampling
    patch_size=patchsize,  # final output dimensions
)
params = {  # For more info on these, see the Forward_Model method
    "stack_depth": 2,
    "psf": {
        "lri": True,
        "stride": 1,
        "symmetric": True,
        "optimize": False,
        "padded_shape": [dim, dim],
        "norm_each": True,
    },
    "operations": {
        "sim_blur": False,
        "sim_meas": True,
        "adjoint": True,
        "spectral_pad": True,
        "roll": False,
    },
}

forward_model = forward.ForwardModel(mask, params, psf_path, device=device)

# The following should show whether we are using a varying or invariant Hfor and Hadj
print(forward_model.fwd, forward_model.adj)

forward.build_data_pairs(
    os.path.join(processed_datapath, "harvard_data"), forward_model
)
forward.build_data_pairs(os.path.join(processed_datapath, "pavia_data"), forward_model)
# forward.build_data_pairs(os.path.join(processed_datapath, "fruit_data"), forward_model)
