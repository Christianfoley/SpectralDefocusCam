# %%
import sys
import os
import glob

sys.path.insert(0, "../../")
sys.path.append("/home/cfoley_waller/defocam/SpectralDefocusCam")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import utils.helper_functions as helper
import train
import numpy as np
import torch

DEVICE = "cuda:2"

TEST_DATA_PATH = [
    "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/spectral_two_point_resolution_volumes",
    "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/spatial_two_point_resolution_volumes",
    "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/vertical_spatial_two_point_resolution_volumes",
]

SAVE_PRED_PATH = [
    "/home/cfoley/defocuscamdata/recons/resolution_analysis/spectral_fista_sparsity",
    "/home/cfoley/defocuscamdata/recons/resolution_analysis/spatial_fista_sparsity",
    "/home/cfoley/defocuscamdata/recons/resolution_analysis/vertical_spatial_fista_sparsity",
]

CONFIG_PATH = "/home/cfoley/SpectralDefocusCam/notebooks/recons_sim_fista/fista_ablation_config_static.yml"

# Set ablation parameters for each experiment
BLURRY_ONLY = [False]  # [False, False, False, False, True]
STACK_DEPTH = [5]  # [1, 2, 3, 5, 1]


def initialize_fista_model(stack_depth, blurry):
    """
    Get the trained model
    """
    config = helper.read_config(CONFIG_PATH)

    config["device"] = torch.device(DEVICE)
    config["forward_model_params"]["operations"]["adj_mask_noise"] = False
    config["forward_model_params"]["operations"]["fwd_mask_noise"] = False
    config["data_precomputed"] = False

    config["forward_model_params"]["stack_depth"] = stack_depth
    exposures = config["forward_model_params"]["psf"]["exposures"]
    if stack_depth == 1:
        stride = -1 if blurry else 1
        exposures = exposures[-1:] if blurry else exposures[:1]
    elif stack_depth == 2:
        stride, exposures = 4, exposures[0::4]
    elif stack_depth == 3:
        stride, exposures = 2, exposures[0::2]
    elif stack_depth == 5:
        stride = 1
    config["forward_model_params"]["psf"]["stride"] = stride
    config["forward_model_params"]["psf"]["exposures"] = exposures

    if blurry and stack_depth == 1:
        config["forward_model_params"]["psf"]["norm"] = "two"  # Because magic!

    config["forward_model_params"]["prox_method"] = "native"
    config["forward_model_params"]["tau"] = 0.1

    model = train.get_model(config=config, device=config["device"])
    model.eval()

    # We also simulate inputs here to allow for 0-1 norming before simulation
    return model.model1, model.model2


def main():
    """Run model inference"""

    for i in range(len(TEST_DATA_PATH)):
        data_files = glob.glob(os.path.join(TEST_DATA_PATH[i], "*.npy"))
        stack_depth = STACK_DEPTH[0]
        blurry = BLURRY_ONLY[0]
        fm, rm = initialize_fista_model(stack_depth, blurry)
        rm.iters = 241
        rm.print_every = 60

        for j, file in enumerate(data_files):
            print(f"{i+1}/{len(STACK_DEPTH)} {j}/{len(data_files)}")
            name = f"pred_fista5_blurry-{blurry}_params-{rm.iters}-{rm.tv_lambda}-{rm.tv_lambdax}-{rm.tv_lambdaw}_{os.path.basename(file)[:-4]}.npy"
            if os.path.exists(os.path.join(SAVE_PRED_PATH[i], name)):
                print(
                    f"Already reconstructed {os.path.basename(file)[:-4]}... Skipping."
                )
                continue

            x = np.load(file).transpose(2, 0, 1)

            ############# Prediction -- must save as numpy file (y,x,lambda) ##########
            sim = fm(torch.tensor(x)[None, None, ...].to(DEVICE))
            _ = rm(sim.squeeze(dim=(0, 2)))
            recon = rm.out_img

            if not os.path.exists(SAVE_PRED_PATH[i]):
                os.makedirs(SAVE_PRED_PATH[i])
            with open(os.path.join(SAVE_PRED_PATH[i], name), "wb") as f:
                np.save(f, recon)


# %%
if __name__ == "__main__":
    main()

# %%
