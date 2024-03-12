import sys, os, glob
import numpy as np
import torch
import scipy.io as io

sys.path.insert(0, "../../")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = "cuda:2"


import dataset.precomp_dataset as ds
import dataset.preprocess_data as prep_data
import utils.helper_functions as helper
import utils.diffuser_utils as diffuser_utils
from models.get_model import get_model

CONFIG_FISTA = "fista_config_static.yml"


def main():
    config = helper.read_config(CONFIG_FISTA)
    model = get_model(config, device=device)
    fm, rm = model.model1, model.model2

    prep_data.preprocess_harvard_data(
        config["base_data_path"],
        config["base_data_path"] + "_preprocessed",
        patch_size=config["patch_crop"],
        skip_masked=False,
    )

    for file in glob.glob(
        os.path.join(config["base_data_path"] + "_preprocessed", "*.mat")
    ):
        # Pipeline: downsamp -> [0,1]norm -> sim -> recon
        img = io.loadmat(file)["image"]
        img = np.stack(
            [
                diffuser_utils.pyramid_down(img[:, :, i], config["patch_size"])
                for i in range(img.shape[-1])
            ]
        )
        gt = torch.tensor(helper.value_norm(img), device=device)[None, None, ...]

        sim = fm(gt).squeeze()
        recon = rm(sim)

        if os.path.exists(config["save_recon_path"]):
            os.mkdir(config["save_recon_path"])

        savename = f"recon_{os.path.basename(file)}{config['image_center']}_{config['patch_size']}_{model.model2.iters}_{model.model2.tv_lambda}_{model.model2.tv_lambdaw}_{model.model2.tv_lambdax}_{model.model2.psfs.shape[0]}.npy"
        with open(os.path.join(config["save_recon_path"], savename), "wb") as f:
            np.save(f, recon)

        gt_savename = f"gt_{os.path.basename(file)}{config['image_center']}_{config['patch_size']}.npy"
        with open(os.path.join(config["save_recon_path"], gt_savename), "wb") as f:
            np.save(f, gt.squeeze().cpu().numpy())


if __name__ == "__main__":
    main()
