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
############################################################################################
#                       Speed comparison for batched ring convolution                      #
############################################################################################
import scipy.io as io
import torch
import numpy as np
from models.rdmpy import blur, deblur
import matplotlib.pyplot as plt
import time, glob
import cv2

# %%
device = torch.device("cuda:0")
print(torch.cuda.get_device_name(device))

# get psf
psf_data = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/10_26/multipos/alignment_psfs_telecent25um_10_26/dim384_3focus/lri_psf_calib_0.mat"
psf_data = torch.tensor(io.loadmat(psf_data)["psf_data"], device=device)


# get sample
sample_dir = "/home/cfoley_waller/defocam/defocuscamdata/sample_data_preprocessed_lri_11_9_sortedmask/fruit_data/"
sample_data = glob.glob(
    os.path.join(
        sample_dir, "internals_artichoke_SegmentedCroppedCompressed_patch_*.mat"
    )
)

img_stack = torch.zeros((len(sample_data), psf_data.shape[0], psf_data.shape[0]))
for i, sample in enumerate(sample_data):
    img = cv2.resize(np.mean(io.loadmat(sample)["image"], 2), (psf_data.shape[0],) * 2)
    img = (img - np.min(img)) / np.max(img - np.min(img))
    img_stack[i] = torch.tensor(img, device=device)


# %%

blurred = blur.batch_ring_convolve(img_stack.unsqueeze(0), psf_data, device=device)[0]

# %%
index = 2  # one-frame deconv indes
opt_params = {"plot_loss": False}

##### processing single image #####
start = time.time()
deblurred = deblur.ring_deconvolve(
    blurred[index], psf_data, device=device, opt_params=opt_params
)
one_frame_time = time.time() - start

#### processing batch #####
start = time.time()
deblurred_batch = deblur.ring_deconvolve(
    blurred, psf_data, device=device, opt_params=opt_params
)
batch_frame_time = time.time() - start

print(f"Single frame time: {one_frame_time:.2f}")
print(f"{img_stack.shape[0]} frame batch time: {batch_frame_time:.2f}")
# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 9), dpi=70)

axes[0].imshow(img_stack[index].cpu().numpy(), cmap="inferno")
axes[0].set_title("Original Image")

axes[1].imshow(blurred[index].cpu().numpy(), cmap="inferno")
axes[1].set_title("Blurred Image")

axes[2].imshow(deblurred, cmap="inferno")
axes[2].set_title("Deblurred Image (alone)")

axes[3].imshow(deblurred_batch[index], cmap="inferno")
axes[3].set_title("Deblurred Image (in batch)")
plt.tight_layout()
plt.show()
# %%
# View residuals for blurred index

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=70)

blur_residual = (img_stack[index].cpu() - blurred[index].cpu()).numpy()
deblur_residual = img_stack[index].cpu().numpy() - deblurred_batch[index]
vmax_val = max(np.max(blur_residual), np.max(deblur_residual)) * 0.5

im1 = axes[0].imshow(blur_residual, cmap="inferno", vmax=vmax_val, vmin=0)
axes[0].set_title(f"Blur residual, {index}")
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(deblur_residual, cmap="inferno", vmax=vmax_val, vmin=0)
axes[1].set_title(f"Deblurred residual, {index}")
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.show()
# %%

# %%
