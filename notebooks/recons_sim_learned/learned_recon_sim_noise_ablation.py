import sys, os, glob, tqdm, pathlib

sys.path.insert(0, "../../")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import dataset.precomp_dataset as ds
import utils.helper_functions as helper
import train
import numpy as np, torch
import scipy.io as io

DEVICE = "cuda:1"

TEST_DATA_PATH = "/home/cfoley/defocuscamdata/recons/model_ablation_test_set"

SAVE_PRED_PATH = (
    "/home/cfoley/defocuscamdata/recons/noise_ablation_maskandshot_preds_learned_classical_unet"
)

# Set ablation parameters for each experiment
MODEL_WEIGHTS = [
    "/home/cfoley/defocuscamdata/models/checkpoint_train_02_07_2024_lsi_adjoint_batchnorm_L1psf_3meas.yml/2024_03_26_15_00_13/saved_model_ep45_testloss_0.043526550366121505.pt",
]
BLURRY_ONLY = [False, False, False, False, False, False]
STACK_DEPTH = [3, ] * 6
NOISE_LEVEL = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]


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
    config["data_precomputed"] = False
    config["preload_weights"] = True
    config["checkpoint_dir"] = trained_weights_path

    # apply noise
    config["forward_model_params"]["operations"]["sim_meas"] = True
    config["forward_model_params"]["operations"]["shot_noise"] = True
    config["forward_model_params"]["operations"]["fwd_mask_noise"] = True

    model = train.get_model(config=config, device=config["device"])
    model.eval()

    return model.model1, model.model2


def main():
    """Run model inference"""
    data_files = glob.glob(os.path.join(TEST_DATA_PATH, "*.mat"))
    fm, rm = initialize_learned_model(0)
    fm.passthrough = False

    for i in range(len(NOISE_LEVEL)):
        stack_depth = STACK_DEPTH[i]
        blurry = BLURRY_ONLY[i]

        for j, file in tqdm.tqdm(
            list(enumerate(data_files))[:200], desc=f"pred{i}"
        ):  # TODO remove!
            fm.mask_noise_intensity = NOISE_LEVEL[i]
            fm.shot_noise_intensity = NOISE_LEVEL[i]

            name = f"pred_learned_{stack_depth}_blurry-{blurry}_noise-{NOISE_LEVEL[i]}_{os.path.basename(file)[:-4]}.npy"
            if os.path.exists(os.path.join(SAVE_PRED_PATH, name)):
                print(
                    f"Already reconstructed {os.path.basename(file)[:-4]}... Skipping."
                )
                continue

            x = helper.value_norm(io.loadmat(file)["image"])

            ############# Prediction -- must save as numpy file (y,x,lambda) ##########
            # for classical unets
            sim = fm.fwd(torch.tensor(x)[None, None, ...].to(DEVICE), fm.psfs, fm.sim_mask_noise(fm.mask))
            sim = (sim - sim.mean()) / sim.std() #normalize sim'd meas
            sim = fm.adj(sim, fm.psfs, fm.mask)
            sim = fm.spectral_pad(sim, 2, 2).to(torch.float32)
            pred = (
                rm(sim)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )
            # for condunets
            # sim = fm(torch.tensor(x)[None, None, ...].to(DEVICE))
            # pred = (
            #     rm((sim - sim.mean()) / sim.std())
            #     .squeeze()
            #     .detach()
            #     .cpu()
            #     .numpy()
            #     .transpose(1, 2, 0)
            # )
            if not os.path.exists(SAVE_PRED_PATH):
                os.makedirs(SAVE_PRED_PATH)
            with open(os.path.join(SAVE_PRED_PATH, name), "wb") as f:
                np.save(f, pred)


if __name__ == "__main__":
    main()
