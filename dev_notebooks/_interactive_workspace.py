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

# %%
# ------------------ testing psf loading ---------------- #

psfs = get_psfs_dmm_37ux178(center_crop_width=256, crop_shape="square", usefirst=True)
for psf in psfs:
    plt.imshow(psf)
    plt.show()
# %%
# ------------------ testing the psf_stack function to be used in forward model ---------------- #
psf_dir = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/psfs_9_22_2023_noiseavg32"
num_ims = 3
mask_shape = (256, 256)
psf_stack = get_psf_stack(psf_dir, num_ims, mask_shape)

for psf in psf_stack:
    plt.imshow(psf)
    plt.show()

# %%
# ------------------ Reading calibration measurements into a matrix ---------------- #
exposures = ["auto", "0.5ms", "1.0ms", "2.0ms"]
for exp in exposures:
    calib_matrix_dir = (
        "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/calib_matrix_9_22_2023_exp"
        + exp
        + "_378-890"
    )
    channs = [
        Image.open(im) for im in glob.glob(os.path.join(calib_matrix_dir, "*.bmp"))
    ]
    matrix = np.stack(channs, axis=0).transpose(1, 2, 0)

    matrix = {"mask": matrix, "dtype": str(matrix.dtype)}
    io.savemat(os.path.join(calib_matrix_dir, "calibration_matrix.mat"), matrix)
# %%
