# %%
import sys

sys.path.insert(0, "/home/cfoley_waller/defocam/SpectralDefocusCam")
import os
import numpy as np
import torch

import models.rdmpy.blur as blur
import models.rdmpy.calibrate as calibrate
import models.forward as fm

import train
import data_utils.dataset as ds
from utils.helper_functions import read_config

# %%
# ----------------------------------------- Specify paths & setup config ------------------------------------------ #
config_path = "/home/cfoley_waller/defocam/SpectralDefocusCam/config_files/training/train_10_18_2023_stack3_exp_NEW_blur_unet.yml"
precomputed_dir = "/home/cfoley_waller/defocam/defocuscamdata/sample_data/precomputed"

config = read_config(config_path)
config["forward_model_params"]["apply_adjoint"] = False
config["forward_model_params"]["spectral_pad_output"] = False
config["forward_model_params"]["sim_blur"] = True
# ----------------------------------------- LOAD DATA & FORWARD MODEL ------------------------------------------ #

train_loader, val_loader, test_loader = ds.get_data(
    1, [0.7, 0.15, 0.15], config["base_data_path"], apply_rand_aug=False
)

model = train.get_model(config, device="cuda:1")
forward_model = model.model1

blur_levels = "_".join(
    ["blur"] + [x[:6] for x in map(str, forward_model.w_init.tolist())]
)
precomputed_dir = os.path.join(precomputed_dir, blur_levels)

# %%

dim = 768
sys_params = {
    "samples": dim,
    "L": 1e-3,
    "lamb": 0.55e-6,
    "pupil_radius": ((dim) * (0.55e-6) * (100e-3)) / (4 * (1e-3)),
    "z": 100e-3,
}
coeffs = np.load(
    "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/10_26/multipos/alignment_psfs_telecent25um_10_26/seidel_coeffs_blur1.npy"
)
device = "cuda:0"

psf_data = calibrate.get_psfs(
    seidel_coeffs=torch.tensor(coeffs, device=device),
    dim=dim,
    model="lri",
    sys_params=sys_params,
    device=device,
    verbose=True,
)
psf_data = psf_data.detach().cpu().numpy()


# %%
# ----------------------------------------- COMPUTE MEASUREMENTS ------------------------------------------ #
train_data_dir = os.path.join(precomputed_dir, "train")
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)

for i, sample in enumerate(train_loader):
    output = forward_model(sample["image"].to("cuda:1")).detach().cpu().numpy()
    np.save(os.path.join(train_data_dir, str(i)), output)


val_data_dir = os.path.join(precomputed_dir, "val")
if not os.path.exists(val_data_dir):
    os.makedirs(val_data_dir)
print(len(val_loader))
for i, sample in enumerate(val_loader):
    output = forward_model(sample["image"].to("cuda:1")).detach().cpu().numpy()
    np.save(os.path.join(val_data_dir, str(i)), output)
    print(i)


test_data_dir = os.path.join(precomputed_dir, "test")
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

for i, sample in enumerate(test_loader):
    output = forward_model(sample["image"].to("cuda:1")).detach().cpu().numpy()
    np.save(os.path.join(test_data_dir, str(i)), output)
# %%


import scipy.io as io
import glob, os
import cv2
from pathlib import Path

pavia_data_path = "/home/cfoley_waller/defocam/defocuscamdata/sample_data/paviadata"
pavia_files = glob.glob(os.path.join(pavia_data_path, "*.mat"))

for file in pavia_files:
    data = io.loadmat(file)
    tag = "pavia"
    if tag not in data:
        tag = "paviaU"
    x, y = data[tag].shape[0], data[tag].shape[1]

    print(f"Shape: {x, y}")

    # get closest multiple of 256
    if abs(x - (x // 256 * 256)) < abs(x - ((x // 256 + 1) * 256)):
        x_mult = x // 256
    else:
        x_mult = x // 256 + 1

    if abs(y - (y // 256 * 256)) < abs(y - ((y // 256 + 1) * 256)):
        y_mult = y // 256
    else:
        y_mult = y // 256 + 1

    for i in range(x_mult):
        for j in range(y_mult):
            chunk_number = i * y_mult + j
            chunk = data[tag][256 * i : 256 * (i + 1), 256 * j : 256 * (j + 1), :]
            print(chunk.shape)
            chunk_file = file[:-4] + f"_chunk_{chunk_number}.mat"
            io.savemat(chunk_file, {tag: chunk})


# %%
