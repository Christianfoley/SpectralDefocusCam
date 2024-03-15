import sys, os, glob, tqdm, pathlib

sys.path.insert(0, "../../")
sys.path.append("/home/cfoley_waller/defocam/SpectralDefocusCam")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import utils.helper_functions as helper
import train
import numpy as np, torch
import scipy.io as io

DEVICE = "cuda:0"

TEST_DATA_PATH = "/home/cfoley_waller/10tb_extension/defocam/defocuscamdata/recons/model_ablation_test_set"

SAVE_PRED_PATH = "/home/cfoley_waller/10tb_extension/defocam/defocuscamdata/recons/model_ablation_test_preds_fista"

CONFIG_PATH = "/home/cfoley_waller/defocam/SpectralDefocusCam/notebooks/recons_sim_fista/fista_ablation_config_static.yml"

# Set ablation parameters for each experiment
BLURRY_ONLY = [False, False]  # [False, False, False, False, True]
STACK_DEPTH = [2, 3]  # [1, 2, 3, 5, 1]


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

    model = train.get_model(config=config, device=config["device"])
    model.eval()

    # We also simulate inputs here to allow for 0-1 norming before simulation
    return model.model1, model.model2


def main():
    """Run model inference"""
    data_files = glob.glob(os.path.join(TEST_DATA_PATH, "*.mat"))

    for i in range(len(STACK_DEPTH)):
        stack_depth = STACK_DEPTH[i]
        blurry = BLURRY_ONLY[i]
        fm, rm = initialize_fista_model(stack_depth, blurry)

        for j, file in enumerate(data_files):
            print(f"{i+1}/{len(STACK_DEPTH)} {j}/{len(data_files)}")
            name = f"pred_fista_{stack_depth}_blurry-{blurry}_params-{rm.iters}-{rm.tv_lambda}-{rm.tv_lambdax}-{rm.tv_lambdaw}_{os.path.basename(file)[:-4]}.npy"
            if os.path.exists(os.path.join(SAVE_PRED_PATH, name)):
                print(
                    f"Already reconstructed {os.path.basename(file)[:-4]}... Skipping."
                )
                continue

            sample = io.loadmat(file)

            ############# Fista requires 0-1 normalization of inputs to simulation model ##########
            x = helper.value_norm(sample["image"])

            ############# Prediction -- must save as numpy file (y,x,lambda) ##########
            sim = fm(torch.tensor(x)[None, None, ...].to(DEVICE))

            _ = rm(sim.squeeze(dim=(0, 2)))
            recon = rm.out_img

            if not os.path.exists(SAVE_PRED_PATH):
                os.makedirs(SAVE_PRED_PATH)
            with open(os.path.join(SAVE_PRED_PATH, name), "wb") as f:
                np.save(f, recon)


if __name__ == "__main__":
    main()
