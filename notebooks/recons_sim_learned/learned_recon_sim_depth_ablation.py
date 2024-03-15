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

SAVE_PRED_PATH = "/home/cfoley/defocuscamdata/recons/model_ablation_test_preds_learned"

# Set ablation parameters for each experiment
MODEL_WEIGHTS = [
    "/home/cfoley/defocuscamdata/models/checkpoint_train_02_07_2024_lsi_adjoint_condunet_L1psf_1meas_blurry.yml/2024_03_12_07_11_06/saved_model_ep6_testloss_0.13591050004305424.pt",
    "/home/cfoley/defocuscamdata/models/checkpoint_train_02_07_2024_lsi_adjoint_condunet_L1psf_1meas_sharp.yml/2024_03_12_07_10_42/saved_model_ep5_testloss_0.06585123277497741.pt",
    "/home/cfoley/defocuscamdata/models/checkpoint_train_02_07_2024_lsi_adjoint_condunet_L1psf_2meas.yml/2024_03_11_14_44_55/saved_model_ep28_testloss_0.04970918206568315.pt",
    "/home/cfoley/defocuscamdata/models/checkpoint_train_02_07_2024_lsi_adjoint_condunet_L1psf_3meas.yml/2024_03_12_00_42_32/saved_model_ep12_testloss_0.05745079041950686.pt",
    "/home/cfoley/defocuscamdata/models/checkpoint_train_02_07_2024_lsi_adjoint_condunet_L1psf_5meas.yml/2024_03_10_23_07_26/saved_model_ep13_testloss_0.10106371374765658.pt",
]
BLURRY_ONLY = [True, False, False, False, False]
STACK_DEPTH = [1, 1, 2, 3, 5]


def initialize_learned_model(index):
    """
    Get the trained model
    """
    trained_weights_path = MODEL_WEIGHTS[index]
    config_path = os.path.join(
        pathlib.Path(trained_weights_path).parent, "training_config.yml"
    )
    config = helper.read_config(config_path)

    _, _, test_loader = ds.get_data_precomputed(
        batch_size=1,
        data_split=config["data_partition"],
        base_path=config["base_data_path"],
        model_params=config["forward_model_params"],
        workers=6,
        shuffle=False,
    )

    config["device"] = torch.device(DEVICE)
    config["forward_model_params"]["operations"]["adj_mask_noise"] = False
    config["forward_model_params"]["operations"]["fwd_mask_noise"] = False
    config["data_precomputed"] = False
    config["preload_weights"] = True
    config["checkpoint_dir"] = trained_weights_path

    model = train.get_model(config=config, device=config["device"])
    model.eval()

    return test_loader, model.model2


def main():
    """Run model inference"""
    # data_files = glob.glob(os.path.join(TEST_DATA_PATH, "*.mat"))

    for i in range(len(MODEL_WEIGHTS)):
        test_loader, model = initialize_learned_model(i)
        stack_depth = STACK_DEPTH[i]
        blurry = BLURRY_ONLY[i]

        for i, sample in tqdm.tqdm(
            list(enumerate(test_loader)), desc=f"preds model {i}"
        ):
            file = test_loader.dataset.file_list[i]
            x = sample["input"]

            ############# Prediction -- must save as numpy file (y,x,lambda) ##########
            pred = (
                model(x.to(DEVICE)).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            )
            name = f"pred_learned_{stack_depth}_blurry-{blurry}_{os.path.basename(file)[:-4]}.npy"
            with open(os.path.join(SAVE_PRED_PATH, name), "wb") as f:
                np.save(f, pred)


if __name__ == "__main__":
    main()
