# %%
import os, sys, glob
import tqdm
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import rotate

sys.path.insert(0, "/home/cfoley_waller/defocam/SpectralDefocusCam")

import utils.psf_calibration_utils as psf_utils
import utils.helper_functions as helper
import utils.diffuser_utils as diffuser_utils
from models.rdmpy._src.util import getCircList

# %%
############### Hyperparameters ################
psf_path = "../defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/10_26/multipos/alignment_psfs_telecent25um_10_26"
crop = (400, 0, 1500, 1000)
focus_levels = 3
patch_size = (768, 768)
alignment_estimate = (880, 421)

coord_method = "conv"
# if coord_method == conv
ksizes = [7, 21, 45]
min_distance = 15
threshold = 0.7

# %%
############### Reading and locating the psfs ################
psfs = psf_utils.read_psfs(psf_path, crop=crop, patchsize=patch_size)
supimp_psfs = psf_utils.superimpose_psfs(psfs, focus_levels=focus_levels, one_norm=True)
# %%
if coord_method == "conv":
    coords = psf_utils.get_psf_coords(
        psfs,
        focus_levels,
        coord_method,
        ksizes,
        threshold,
    )
else:
    coords = psf_utils.get_psf_coords(
        supimp_psfs,
        focus_levels,
        coord_method,
        threshold=threshold,
        min_distance=min_distance,
    )
# %%
for psf_set_num in range(0, 16):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    fig.set_dpi(60)
    cropim = lambda im, coords, size: im[
        coords[0] - size // 2 : coords[0] + size // 2,
        coords[1] - size // 2 : coords[1] + size // 2,
    ]
    for i in range(3):
        ax[i].imshow(cropim(psfs[psf_set_num * 3 + i], coords[i][psf_set_num], 64))

    plt.show()
# %%
############### Getting radial image subdivisions ################

cropped_align_est = (alignment_estimate[0] - crop[0], alignment_estimate[1] - crop[1])
maxval = diffuser_utils.get_radius(patch_size[0] // 2, patch_size[1] // 2)
subdivisions, radii = psf_utils.radial_subdivide(
    coords, cropped_align_est, maxval, True
)


# %%
############### Plotting subdivisions for each focus level ################

fig, ax = plt.subplots(1, focus_levels, figsize=(8 * focus_levels, 5))
fig.set_dpi(70)
for i in range(focus_levels):
    ax[i].scatter(radii[i], range(len(coords[i])))
    ax[i].set_xlabel("radius")
    ax[i].set_ylabel("order")
    ax[i].set_title(f"focus {i}")

    ax[i].vlines(
        subdivisions[i],
        ymin=0,
        ymax=len(subdivisions[i]),
        colors=["red"] * len(subdivisions),
    )
plt.suptitle(
    "PSF distances from alignment axis and chosen radial subdivisions", fontsize=14
)
plt.show()
# %%
############### Getting rotated psf rings ################
level = 2
rotated_psfs = psf_utils.view_patched_psf_rings(
    psfs[level::focus_levels], coords[level], dim=768
)

plt.imshow(rotated_psfs, cmap="inferno", interpolation="none")
plt.show()

# %%
############### Hyperparameters ################
import utils.psf_calibration_utils as psf_utils
import torch

psf_path = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/11_21/multipos/alignment_psfs_cross_5_blur"
blur_levels = 5
crop_size = 256
dim = 256
alignment_estimate = (1000, 472)

coord_method = "conv"
ksizes = [7, 21, 45, 55, 65]
exposures = [1 / 662, 1 / 110, 1 / 45, 1 / 30, 1 / 21]
min_distance = 25
threshold = 0.7
psf_dim = 300
polar = True
device = torch.device("cuda:0")
torch.cuda.get_device_name(device)
# %%
psf_data = psf_utils.get_lri_psfs(
    psf_path,
    blur_levels,
    crop_size,
    dim,
    alignment_estimate,
    coord_method,
    ksizes,
    exposures,
    min_distance,
    threshold,
    psf_dim,
    polar=polar,
    device=device,
    verbose=True,
)

# %%
print(psf_data.shape)
import matplotlib.pyplot as plt
import numpy as np

if polar:
    plt.imshow(np.mean(np.real(psf_data[0, ...]), axis=1))
else:
    plt.imshow(psf_data[4, 383, :, :] + psf_data[0, 100, :, :])
plt.colorbar()
plt.show()
# %%
