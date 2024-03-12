import sys, os, glob, tqdm, pathlib

sys.path.insert(0, "../../")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import utils.helper_functions as helper
import train
import numpy as np, torch
import scipy.io as io

DEVICE = "cuda:2"

TEST_DATA_PATH = "/home/cfoley_waller/10tb_extension/defocam/defocuscamdata/recons/model_ablation_test_set"

SAVE_PRED_PATH = "/home/cfoley_waller/10tb_extension/defocam/defocuscamdata/recons/model_ablation_test_preds_learned"

# Set ablation parameters for each experiment
MODEL_WEIGHTS = [
    "/home/cfoley_waller/defocam/defocuscamdata/models/checkpoint_train_02_07_2024_lsi_adjoint_condunet_L1psf_2meas.yml/2024_03_11_14_44_55/saved_model_ep28_testloss_0.04970918206568315.pt",
    "/home/cfoley_waller/defocam/defocuscamdata/models/checkpoint_train_02_07_2024_lsi_adjoint_condunet_L1psf_3meas.yml/2024_03_12_00_42_32/saved_model_ep8_testloss_0.06279138345622791.pt",
]
BLURRY_ONLY = [False, False]
STACK_DEPTH = [2, 3]


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
    return model.model2


def main():
    """Run model inference"""
    data_files = glob.glob(os.path.join(TEST_DATA_PATH, "*.mat"))

    for i in range(len(MODEL_WEIGHTS)):
        model = initialize_learned_model(i)
        stack_depth = STACK_DEPTH[i]
        blurry = BLURRY_ONLY[i]

        for file in data_files:
            sample = io.loadmat(file)

            ############# Slicing Depends on stack depth of model ##########
            if stack_depth == 1:
                if blurry:
                    x = sample["input"][4]  # 4
                else:
                    x = sample["input"][0]  # 0
            elif stack_depth == 2:
                x = sample["input"][0::4]  # 0, 4
            elif stack_depth == 3:
                x = sample["input"][0::2]  # 0, 2, 4
            elif stack_depth == 5:
                x = sample["input"]  # 0, 1, 2, 3, 4

            ############# Prediction -- must save as numpy file (y,x,lambda) ##########
            pred = (
                model(torch.tensor(x)[None, ...].to(DEVICE))
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )
            name = f"pred_learned_{stack_depth}_blurry-{blurry}_{os.path.basename(file)[:-4]}.npy"
            with open(os.path.join(SAVE_PRED_PATH, name), "wb") as f:
                np.save(f, pred)


if __name__ == "__main__":
    main()
