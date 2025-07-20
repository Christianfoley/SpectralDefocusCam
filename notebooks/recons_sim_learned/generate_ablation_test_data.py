import sys

sys.path.insert(0, "../..")

import dataset.precomp_dataset as ds
import utils.helper_functions as helper
import scipy.io as io
import os
import tqdm

# BASE_PATH is the path to the dataset split in exactly the same way as below used to train every model
BASE_PATH = "/home/cfoley_waller/10tb_extension/defocam/defocuscamdata/sample_data_preprocessed/sample_data_preprocessed_lsi_02_07_L1psf_5meas"

# STORE_PATH is the place where we will extract all of the test set samples to
STORE_PATH = "/home/cfoley_waller/10tb_extension/defocam/defocuscamdata/recons/model_ablation_test_set"

# CONFIG is the path to the static learned model config for this ablation
CONFIG = "learned_config_static.yml"


def main():
    config = helper.read_config(CONFIG)

    train_loader, val_loader, test_loader = ds.get_data_precomputed(
        batch_size=config["batch_size"],
        data_split=config["data_partition"],
        base_path=BASE_PATH,
        model_params=config["forward_model_params"],
        norm_target=False,
        shuffle=False,
        workers=6,
    )

    for i, sample in tqdm.tqdm(enumerate(test_loader.dataset)):
        name = os.path.basename(test_loader.dataset.file_list[i])
        io.savemat(os.path.join(STORE_PATH, name), sample)


if __name__ == "__main__":
    main()
