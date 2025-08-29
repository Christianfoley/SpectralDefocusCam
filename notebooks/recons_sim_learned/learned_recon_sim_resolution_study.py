import sys
import os
import glob
import tqdm
import pathlib

sys.path.insert(0, "../../")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import utils.helper_functions as helper
import train
import numpy as np
import torch

DEVICE = "cuda:1"

TEST_DATA_PATH = [
    "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/spectral_two_point_resolution_volumes",
    "/home/emarkley/Workspace/PYTHON/SpectralDefocusCam/notebooks/eric_simulation_studies/spatial_two_point_resolution_volumes",
]

SAVE_PRED_PATH = [
    "/home/cfoley/defocuscamdata/recons/resolution_analysis/spectral_learned",
    "/home/cfoley/defocuscamdata/recons/resolution_analysis/spatial_learned",
]

# Set ablation parameters for each experiment
MODEL_WEIGHTS = [
    "/home/cfoley/defocuscamdata/models/checkpoint_train_02_07_2024_lsi_adjoint_condunet_L1psf_2meas.yml/2024_03_11_14_44_55/saved_model_ep28_testloss_0.04970918206568315.pt",
]
BLURRY_ONLY = [False]
STACK_DEPTH = [2]


def initialize_learned_model(index):
    """
    Get the trained model
    """
    trained_weights_path = MODEL_WEIGHTS[index]
    config_path = os.path.join(
        pathlib.Path(trained_weights_path).parent, "training_config.yml"
    )
    config = helper.read_config(config_path)

    config["device"] = torch.device(DEVICE)
    config["forward_model_params"]["operations"]["adj_mask_noise"] = False
    config["forward_model_params"]["operations"]["fwd_mask_noise"] = False
    config["data_precomputed"] = False
    config["preload_weights"] = True
    config["checkpoint_dir"] = trained_weights_path

    model = train.get_model(config=config, device=config["device"])
    model.eval()

    return model.model1, model.model2


def main():
    """Run model inference"""
    fm, rm = initialize_learned_model(0)
    fm.passthrough = False

    for i in range(len(TEST_DATA_PATH)):
        data_files = glob.glob(os.path.join(TEST_DATA_PATH[i], "*.npy"))
        stack_depth = STACK_DEPTH[0]
        blurry = BLURRY_ONLY[0]

        for j, file in tqdm.tqdm(list(enumerate(data_files)), desc=f"pred{i}"):
            name = f"pred_learned_{stack_depth}_blurry-{blurry}_{os.path.basename(file)[:-4]}.npy"
            if os.path.exists(os.path.join(SAVE_PRED_PATH[i], name)):
                print(
                    f"Already reconstructed {os.path.basename(file)[:-4]}... Skipping."
                )
                continue
            x = np.load(file).transpose(2, 0, 1)

            ############# Prediction -- must save as numpy file (y,x,lambda) ##########
            sim = fm(torch.tensor(x)[None, None, ...].to(DEVICE))
            pred = (
                rm((sim - sim.mean()) / sim.std())
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )
            with open(os.path.join(SAVE_PRED_PATH[i], name), "wb") as f:
                np.save(f, pred)


if __name__ == "__main__":
    main()
