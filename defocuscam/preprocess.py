import os

import ray
import torch


import defocuscam.models.forward as forward
import defocuscam.dataset.preprocess_data as prep_data
import defocuscam.utils.helper_functions as helper

from defocuscam.models.get_model import get_model


def run_preprocessing(source_path, dest_path, patch_size, overwrite=True, depth=30):
    """
    Runs preprocessing on raw data given the base dir of the raw data and
    the base dir of a destination.

    NOTE: Dangerous; will overwrite data in destination path by default

    Parameters
    ----------
    source_path : str
        path to parent dir of raw data
    dest_path : str
        path to empty destination parent dir for preprocessed data
    patch_size : tuple(int, int)
        xy patch size for images
    overwrite : bool
        whether to allow overwriting data in destination path
    """
    assert dest_path is not source_path, "Destination path must not be source path..."
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    else:
        if not overwrite:
            print(f"Path exists {dest_path}, stopping...")
            return

    harv_source = os.path.join(source_path, "harvard")
    harv_dest = os.path.join(dest_path, "harvard_data")
    prep_data.preprocess_harvard_data(harv_source, harv_dest, patch_size, depth)

    pav_source = os.path.join(source_path, "paviadata")
    pav_dest = os.path.join(dest_path, "pavia_data")
    prep_data.preprocess_pavia_data(pav_source, pav_dest, patch_size, depth)

    fruit_source = os.path.join(source_path, "fruitdata/pca")
    fruit_dest = os.path.join(dest_path, "fruit_data")
    prep_data.preprocess_fruit_data(fruit_source, fruit_dest, patch_size, depth)

    icvl_source = os.path.join(source_path, "icvldata")
    icvl_dest = os.path.join(dest_path, "icvl_data")
    prep_data.preprocess_icvl_data(icvl_source, icvl_dest, patch_size, depth)


def run_precomputation(config, device, batch=1):
    """
    Precomputes training pairs, placing them back in the preprocessed data path
    with their relevent ground truth sample.

    Parameters
    ----------
    config : dict
        config dictionary
    device : torch.cuda.device
        device to use
    batch : int, optional
        batch size for computing training pairs, by default 8
    """
    device = helper.get_device(config["device"])

    fm = get_model(config, device, fwd_only=True)
    fm.passthrough = False  # manually set this to activate the forward model here
    prepd_data = config["base_data_path"]

    forward.build_data_pairs(os.path.join(prepd_data, "harvard_data"), fm, batch)
    forward.build_data_pairs(os.path.join(prepd_data, "pavia_data"), fm, batch)
    forward.build_data_pairs(os.path.join(prepd_data, "fruit_data"), fm, batch)
    forward.build_data_pairs(os.path.join(prepd_data, "icvl_data"), fm, batch)


def main(config, num_cpus=8):
    # setup device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    ray.init(num_cpus=num_cpus)

    print("Num devices: ", torch.cuda.device_count())
    device = helper.get_device(config["device"])
    try:
        print("Trying device: ", torch.cuda.get_device_properties(device).name)
        device = torch.device(device)
    except Exception as e:
        print(f"Failed to select device {device}: {e}")
        print("Running on CPU")
        device = "cpu"

    # run preprocessing from raw data
    source_data_path = config.get(
        "source_data_path",
        "data/sample_data/",
    )

    run_preprocessing(
        source_data_path,
        config["base_data_path"],
        config["patch_size"],
        config["recon_model_params"].get("spectral_depth", 30),
    )

    # precompute training pairs from preprocessed data
    if config["passthrough"]:
        run_precomputation(config, device, batch=config["batch_size"])
