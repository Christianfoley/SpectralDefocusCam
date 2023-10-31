# %%
import sys
import matplotlib.pyplot as plt
import os
import glob
import PIL.Image as Image
import numpy as np
import scipy.io as io

sys.path.insert(0, "/home/cfoley_waller/defocam/SpectralDefocusCam")

from utils.psf_calibration_utils import get_psfs_dmm_37ux178, get_psf_stack
import utils.psf_calibration_utils as psf_utils

# %%
path = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/10_26/multipos/alignment_psfs_telecent25um_10_26"
focus_levels = 3  # levels of focus in your measurements
estimate = (
    "median"  # mean if your points seem to shift very consistently, median if they dont
)
crop = (200, 200, 1400, 1400)  # we will limit our crop to this region


center_estimate = psf_utils.estimate_alignment_center(
    path, focus_levels, estimate="median", crop=crop, verbose=True
)
print(center_estimate)
# %%
# ------------------ testing psf loading ---------------- #
psf_dir = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/psfs_selected_10_18_2023_new_noiseavg32"
psfs = get_psfs_dmm_37ux178(
    psf_dir=psf_dir,
    center_crop_width=256,
    center_crop_shape="square",
    usefirst=True,
    threshold=False,
)
for psf in psfs:
    plt.imshow(psf)
    plt.show()

# %%
# ------------------ testing the psf_stack function to be used in forward model ---------------- #
num_ims = 3
mask_shape = (30, 256, 256)
padded_shape = (768, 768)
blurstride = 1
psf_stack = get_psf_stack(psf_dir, num_ims, mask_shape, padded_shape, blurstride)

for psf in psf_stack:
    plt.imshow(psf)
    plt.show()

# %%
# ------------------ Reading calibration measurements into a matrix ---------------- #
exposures = ["16.67ms", "25.64ms"]
for exp in exposures:
    calib_matrix_dir = (
        "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/calib_matrix_full_10_18_2023_NEW_exp"
        + exp
        + "_gain14.9_navg16_378-890"
    )
    channs = [
        Image.open(im) for im in glob.glob(os.path.join(calib_matrix_dir, "*.bmp"))
    ]
    matrix = np.stack(channs, axis=0).transpose(1, 2, 0)

    matrix = {"mask": matrix, "dtype": str(matrix.dtype)}
    io.savemat(os.path.join(calib_matrix_dir, "calibration_matrix.mat"), matrix)
# %%
#### Try denoising an average psf stack ###

psf_dir = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/"
pos_dirs = [
    "psfs_9_25_2023_pos1",
    "psfs_9_25_2023_pos2",
    "psfs_9_25_2023_pos3",
    "psfs_9_25_2023_pos4",
    "psfs_9_25_2023_pos5",
]
psf_stacks = []
num_ims = 8
mask_shape = (1, 480, 640)

for pos in pos_dirs:
    psf_stack = get_psf_stack(
        os.path.join(psf_dir, pos), num_ims, mask_shape, blurstride=1
    )
    psf_stacks.append(psf_stack)

psf_stack = np.mean(np.stack(psf_stacks, 0), axis=0)
for i, psf in enumerate(psf_stack):
    plt.imshow(psf)
    plt.show()

    psf_im = Image.fromarray(psf)
    psf_im.save(
        f"/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/psfs_9_25_2023_avg_pos/Image000{i+1}.tiff"
    )
# %%
shift_calib_path = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/calibrate shift"

images = [
    np.array(Image.open(im), dtype=float)
    for im in sorted(glob.glob(shift_calib_path + "/*.bmp"))
]
# %%
### Shift calibration workspace ###
for i in range(0, 3):
    plt.imshow(images[i][550:650, 180:280])
    plt.show()

for i in range(3, 6):
    plt.imshow(images[i][35:135, -510:-410])
    plt.show()

fig, ax = plt.subplots(1, 3, figsize=(20, 7.5))
ax[0].imshow(np.sum(np.stack(images, 0), 0))
ax[1].imshow((images[0] + images[1] + images[2])[550:650, 180:280])
ax[2].imshow((images[3] + images[4] + images[5])[35:135, -510:-410])
ax[0].set_title("All psfs overlaid, shown in uncropped FoV")
ax[1].set_title(f"Overlaid PSFs middle left. Offset: {[550,180]}")
ax[2].set_title(f"Overlaid PSFs top right. Offset: {[35,3072 - 510]}")

plt.suptitle(
    "Demonstration of off-axis psf translation: 3 levels of focus/position \nSource: phone flashlight pointed towards camera through pinhole",
    fontsize=28,
)
plt.show()
# %%
