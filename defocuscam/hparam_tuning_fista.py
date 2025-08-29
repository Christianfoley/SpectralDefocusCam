import os
import glob
import tqdm
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import PIL.Image as Image

import defocuscam.utils.helper_functions as helper
import defocuscam.utils.diffuser_utils as diffuser_utils
from defocuscam import train


def fista_hparam_search(
    model,
    measurement,
    tv_lambda_range=(0.005, 1),
    num_lambda=10,
    base=2,
    searchbase=1.4,
    search_num=3,
    num_iters=300,
    savedir="",
):
    """
    Performs a lambda hyperparameter search on fista TV parameters. Varies the base lambda,
    and searches for lambda x,y, and w weighting values relative to each TV lambda tried.

    Lambda range is logscale, and search_num x and w weights are tried around each lambda value
    on logscale with base logscale_base.
    Total number of reconstructions run is lambda_num * (search_num * 2 - 1) ^ (2 (for x and w)).

    Example:
        tv_lambda = 0.1
        tv_lambdax values to try: [0.25 0.5 1. 2. 4.]
        tv_lambdaw values to try: [0.25 0.5 1. 2. 4.]

        Total values tried = 5 * 5 * 1 = 25

    Parameters
    ----------
    model : torch.nn.Module
        fista model
    measurement : torch.tensor
        measurement to run reconstruction hparam search on.
    tv_lambda_range : tuple, optional
        lower and upper bounds of logrange for lambda search, by default (0.005, 1)
    num_lambda : int, optional
        number of lambda values to try
    base : int or str, optional
        base of logscale for lambda AND local weight params, by default 2
    search_num : int, optional
        number of local weight params around each lambda value tried, by default 2
    num_iters : int, optional
        number of fista iters, by default 300
    savedir : str, optional
        optional location to save output figures
    """
    logbase = {1: lambda x: x, 2: np.log2, 10: np.log10, "e": np.log}
    assert base in logbase, "Base not supported"
    log = logbase[base]
    start, end = tv_lambda_range
    recon_progress = model.model2.show_recon_progress

    model.model2.iters = num_iters
    model.model2.show_recon_progress = False
    model.model2.break_diverge_early = True

    # get hyperparameter value grid
    tv_l_vals = np.logspace(log(start), log(end), num=num_lambda, base=base)
    x_w_vals = np.zeros((num_lambda, search_num * 2 - 1, search_num * 2 - 1, 2))

    for i, val in enumerate(tv_l_vals):
        startval, endval = (-search_num + 1, search_num - 1)
        search_values = np.logspace(startval, endval, search_num * 2 - 1, base=searchbase)
        x_w_vals[i] = np.stack(np.meshgrid(search_values, search_values), axis=2)

    # traverse grid
    best = [None, np.inf]
    for v in range(x_w_vals.shape[0]):
        for i in range(x_w_vals.shape[1]):
            for j in range(x_w_vals.shape[2]):
                model.model2.tv_lambda = tv_l_vals[v]
                model.model2.tv_lambdax = x_w_vals[v, i, j, 0]
                model.model2.tv_lambdaw = x_w_vals[v, i, j, 1]

                recon = np.squeeze(model(measurement).get())

                if np.min(model.model2.llist) < np.min(best[1]):
                    best = [(tv_l_vals[v], x_w_vals[v, i, j]), model.model2.llist]

                if savedir:
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    fig = plt.figure(facecolor="white", figsize=(9, 9), dpi=100)
                    plt.imshow(
                        (helper.value_norm(helper.stack_rgb_opt_30(recon)) * 255).astype(np.uint8)
                    )
                    imgfile = f"lambda{tv_l_vals[v]:.3f}_x{x_w_vals[v, i, j, 0]:.3f}_w{x_w_vals[v, i, j, 1]:.3f}_loss{model.model2.llist[-1]}.png"
                    plt.savefig(os.path.join(savedir, str(imgfile)))
                print(
                    f"Lambda: {tv_l_vals[v]:.4f} --- {i}/{x_w_vals.shape[1]} {j}/{x_w_vals.shape[2]}: Loss = {np.min(model.model2.llist):.4f}/{np.min(best[1]):.4f}"
                )

    model.model2.show_recon_progress = recon_progress
    return best


def main():
    device = "cuda:1"
    psf_path_rel = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/11_21/singlepos/psfs_ONAXIS_telecent25um"
    test_meas_path_rel = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/11_21/exp_meas/duckincar"
    calib_mat_path_rel = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/calib_matrix_11_10_2023_2_preprocessed/calibration_matrix_450-810_30chan_stride12_avg12.mat"

    # exp params
    w_init = [0.003, 0.018, 0.04, 0.06, 0.095]
    crop_center = [1000, 2000]
    crop_size = [768, 768]
    patch_size = [384, 384]
    exposures = [1 / 662, 1 / 110, 1 / 45, 1 / 30, 1 / 21]
    num_ims = len(exposures)

    files = sorted(glob.glob(os.path.join(test_meas_path_rel, "*.bmp")))
    prep = lambda x: diffuser_utils.preprocess_meas(
        x,
        center=crop_center,
        crop_size=crop_size[0],
        dim=patch_size[0],
        outlier_std_threshold=3,
    )
    measurements = [prep(np.array(Image.open(x))) for x in tqdm.tqdm(files, "Preprocessing")]

    # ----------- Build model ---------------#
    stack_depth = num_ims  # number of images in your stack
    blurstride = 1  # stride between ordered blur levels of your measurements
    config = {
        "device": device,
        "mask_dir": calib_mat_path_rel,
        "psf_dir": psf_path_rel,
        "data_precomputed": False,
        "forward_model_params": {
            "stack_depth": stack_depth,
            "psf": {
                "lri": False,
                "stride": blurstride,
                "symmetric": True,
                "optimize": False,
                "padded_shape": crop_size[0],
                "exposures": exposures,
                "w_init": w_init,
            },
            "operations": {
                "sim_blur": False,
                "sim_meas": False,
                "adjoint": False,
                "spectral_pad": False,
                "roll": False,
            },
        },
        "recon_model_params": {
            "model_name": "fista",
            "iters": 300,
            "prox_method": "tv",
            "tau": 0.5,
            "tv_lambda": 0.0005,
            "tv_lambdaw": 0.0005,
            "tv_lambdax": 0.0005,
            "lowrank_lambda": 0.05,
            "print_every": 1,
        },
        "batch_size": 1,
        "patch_size": patch_size,
        "patch_crop": crop_size,
        "image_center": crop_center,
        "loss_function": {"name": "mse", "params": {}},
    }

    sel_meas_stack = torch.tensor(np.stack(measurements, axis=0))[
        : stack_depth * blurstride : blurstride
    ]
    model = train.get_model(config, device=device)

    # ---------- Run hyperparameter tuning ------------ #
    print("\nRunning hyperparameter search:")
    start = time.time()
    measurement = sel_meas_stack.unsqueeze(0).to(device)
    savedir = os.path.join(
        "/home/cfoley_waller/defocam/defocuscamdata/fista_hparam_search",
        helper.get_now(),
    )
    best_w, llist = fista_hparam_search(
        model,
        measurement,
        (2, 10),
        num_lambda=5,
        base=2,
        searchbase=1.6,
        search_num=3,
        num_iters=300,
        savedir=savedir,
    )

    print(f"Best params: \n\t tv_lambda: {best_w[0]} \n\t x and y weights: {best_w[1]} \n\t")
    print(f"Minimal loss: {np.min(llist)}")
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
