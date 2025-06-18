import sys

sys.path.insert(0, "../..")

import dataset.precomp_dataset as ds
import scipy.io as io
import os
import tqdm

# Path to the dataset split in exactly the same way as below used to train every model
BASE_PATH = "/home/cfoley/defocuscamdata/sample_data_preprocessed/sample_data_preprocessed_lsi_02_07_L1psf_5meas"

# Out paths for the different test set partitions
FULL_TEST_DATASET_OUT_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "data", "test_set_full"
)
SMALL_TEST_DATASET_OUT_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "data", "test_set_small"
)

# Parameters from the forward model used to preprocess the dataset at BASE_PATH, which
# are used to build a unique key to verify the integrity of the preprocessed dataset
CONFIG = {  # from "learned_config_static.yml"
    "batch_size": 2,
    "data_partition": [0.7, 0.15, 0.15],
    "forward_model_params": {
        "stack_depth": 5,
        "psf": {
            "lri": False,
            "stride": 1,
            "symmetric": True,
            "optimize": False,
            "padded_shape": 768,
            "largest_psf_diam": 128,
            "exposures": [0.00151, 0.00909, 0.02222, 0.03333, 0.04761],
            "threshold": 0.55,
            "norm": "one",
        },
        "operations": {
            "sim_blur": False,
            "sim_meas": True,
            "adjoint": False,
            "spectral_pad": False,
            "adj_mask_noise": False,
            "fwd_mask_noise": True,
        },
        "mask_noise": {
            "intensity": [0, 0.03],
            "stopband_only": True,
            "type": "gaussian",
        },
    },
}


def main():
    _, _, test_loader = ds.get_data_precomputed(
        batch_size=CONFIG["batch_size"],
        data_split=CONFIG["data_partition"],
        base_path=BASE_PATH,
        model_params=CONFIG["forward_model_params"],
        norm_target=False,
        shuffle=False,
        workers=6,
    )

    # Save off full test dataset #848 samples
    os.makedirs(FULL_TEST_DATASET_OUT_DIR, exist_ok=True)
    for i, sample in tqdm.tqdm(enumerate(test_loader.dataset)):
        name = os.path.basename(test_loader.dataset.file_list[i])
        io.savemat(os.path.join(FULL_TEST_DATASET_OUT_DIR, name), sample)

    # Save off small test dataset (84 samples)
    os.makedirs(SMALL_TEST_DATASET_OUT_DIR, exist_ok=True)
    for i, sample in tqdm.tqdm(enumerate(test_loader.dataset)):
        if i % 10 != 0:
            continue
        name = os.path.basename(test_loader.dataset.file_list[i])
        io.savemat(os.path.join(SMALL_TEST_DATASET_OUT_DIR, name), sample)


if __name__ == "__main__":
    main()
