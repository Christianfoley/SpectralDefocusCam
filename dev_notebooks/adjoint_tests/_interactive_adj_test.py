# %%
import sys
import os
import torch
import time

sys.path.insert(0, "/home/cfoley_waller/defocam/SpectralDefocusCam")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import models.forward as forward
import utils.diffuser_utils as diffuser_utils
import utils.helper_functions as helper

# %%
############### LOAD MODEL CONFIG ###################
config_path = "/home/cfoley_waller/defocam/SpectralDefocusCam/config_files/training/train_11_9_2023_lri_precomputed_stack3.yml"
config = helper.read_config(config_path)
device = helper.get_device(config["device"])

# turn off passthrough if true
config["data_precomputed"] = False

# %%
############### INITIALIZE MODEL FROM CONFIG PARAMETERS ###################

mask = diffuser_utils.load_mask(
    path=config["mask_dir"],
    patch_crop_center=config["image_center"],
    patch_crop_size=config["forward_model_params"]["psf"]["padded_shape"],
    patch_size=config["patch_size"],
)
forward_model = forward.ForwardModel(
    mask,
    config["forward_model_params"],
    config["psf_dir"],
    device=device,
)
forward_model.init_psfs()

Hfor, Hadj = forward_model.Hfor, forward_model.Hadj
if config["forward_model_params"]["psf"]["lri"]:
    Hfor, Hadj = forward_model.Hfor_varying, forward_model.Hadj_varying

print(forward_model.psfs.shape, forward_model.mask.shape)
# %%
############### COMPUTE FORWARD AND ADJOINT COMPONENTS ###################
sd = config["forward_model_params"]["stack_depth"]
x = torch.rand(size=(1, sd, 30, 256, 256)).to(device=device)
y = torch.rand(size=(1, sd, 1, 256, 256)).to(device=device)

start = time.time()
y_tilde = Hfor(x, forward_model.psfs, forward_model.mask).float()
x_tilde = Hadj(y, forward_model.psfs, forward_model.mask).float()

print(f"Forward + Adjoint time, batch {x.shape[0]}: {time.time()-start:.2f}s")
# %%
############### DOT-PRODUCT TEST CLOSENESS ###################

y_dot = (y.ravel()).dot(y_tilde.ravel())
x_dot = (x.ravel()).dot(x_tilde.ravel())
rtol = 0.0015

print((y.ravel()).dot(y_tilde.ravel()))
print((x.ravel()).dot(x_tilde.ravel()))

assert torch.isclose(
    y_dot, x_dot, rtol=rtol
), f"Adjoint test with tolerance {rtol} failed"


# %%
