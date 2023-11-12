# %%
import sys
import matplotlib.pyplot as plt
import os
import glob
import PIL.Image as Image
import numpy as np
import scipy.io as io
import tqdm

#
sys.path.insert(0, "/home/cfoley_waller/defocam/SpectralDefocusCam")
# %%
from utils.psf_calibration_utils import get_psfs_dmm_37ux178, get_lsi_psfs
import utils.psf_calibration_utils as psf_utils

# %%
io.loadmat(
    "/home/cfoley_waller/defocam/defocuscamdata/sample_data_preprocessed_11_2/pavia_data/PaviaCenter_bot_patch_0.mat"
).keys()
# %%
import torch
import utils.helper_functions as helper
import data_utils.precomp_dataset as pre_ds

# setup device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = "/home/cfoley_waller/defocam/SpectralDefocusCam/config_files/training/train_11_1_2023_lri_precomputed.yml"
config = helper.read_config(config)

print("Num devices: ", torch.cuda.device_count())
device = helper.get_device(config["device"])

train_loader, val_loader, _ = pre_ds.get_data_precomputed(
    1,
    config["data_partition"],
    config["base_data_path"],
    0,
    config["forward_model_params"],
)

# %%
for sample in train_loader:
    y, x = sample["image"], sample["input"]
    print(y.shape, x.shape)
    break
# %%
fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.mean(y.numpy()[0], 0))
ax[1].imshow(x.numpy()[0, 0, 5])
# %%

helper.plt3D(x.numpy()[0, 0].transpose(1, 2, 0), size=(3, 3))
# %%
import scipy.io as io

mat = io.loadmat(
    "/home/cfoley_waller/defocam/defocuscamdata/sample_data_preprocessed/harvard_data/imgf5_patch_15.mat"
)
a = mat["image"]
print(a.shape)

plt.imshow(np.mean(a, 2))
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
        Image.open(im)
        for im in sorted(glob.glob(os.path.join(calib_matrix_dir, "*.bmp")))
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
