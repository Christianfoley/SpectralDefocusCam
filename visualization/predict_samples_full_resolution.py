import sys

sys.path.append("../")

import numpy as np

import os
import tqdm
import scipy.io
import json

from models.ensemble import SSLSimulationModel
from utils import helper_functions as helper
from train import get_model

from studies.simulation_methods_comparison_collage import patch_predict_utils


SIM_GT_DIRECTORY = (
    "/home/cfoley/SpectralDefocusCam/visualization/data/icvl_double_largepatch_no_pairs"
)
OUT_DIR = "/home/cfoley/SpectralDefocusCam/visualization/results"
SIM_SAMPLES = [
    ("BGU_0522-1217_patch_0.mat", 0.5, (380, 660)),
    ("bulb_0822-0909_patch_0.mat", 0.5, (380, 660)),
    ("Lehavim_0910-1622_patch_0.mat", 0.5, (380, 660)),
    ("BGU_0403-1419-1_patch_0.mat", 0.5, (380, 660)),
    ("eve_0331-1602_patch_0.mat", 0.5, (380, 660)),
    ("gavyam_0823-0930_patch_0.mat", 0.6, (360, 660)),
    ("grf_0328-0949_patch_0.mat", 0.5, (380, 660)),
    ("plt_0411-1155_patch_0.mat", 0.54, (360, 660)),
]
SIM_CROP_SIZE = (420 * 2, 620 * 2)
SIM_PATCH_SIZE = (420, 620)
FISTA_ITERATIONS_BETWEEN_PRINT = 10
FISTA_PLOT_WITH_PRINT = True
FISTA_ITERS = 501

FISTA_CONFIG_PATH = "/home/cfoley/SpectralDefocusCam/studies/simulation_methods_comparison_collage/configs/defocuscam_fista.yml"
LEARNED_CONFIG_PATH = "/home/cfoley/SpectralDefocusCam/studies/simulation_methods_comparison_table/configs/defocuscam_learned.yml"

# ---------------- Code for generating sample dataset ------------------ #
# ray.init(num_cpus=8)
# try:
#     icvl_data = prep_data.preprocess_icvl_data(
#         datapath="/home/cfoley/defocuscamdata/sample_data/icvldata",
#         outpath="/home/cfoley/SpectralDefocusCam/visualization/data/icvl_double_largepatch_no_pairs",
#         patch_size=(
#             learned_bigpatch_config["patch_size"][0] * 2,
#             learned_bigpatch_config["patch_size"][1] * 2,
#         ),
#         overwrite_existing=False,
#     )
# except Exception as e:
#     logging.error(f"Error preprocessing ICVL data: {e}", exc_trace=True)
#     icvl_data = []
# finally:
#     ray.shutdown()


def get_fista_model(
    config_path: str,
    device: str,
    fista_lr_mult=1.0,
) -> SSLSimulationModel:
    """Boilerplate additional steps when loading a model with FISTA recons"""
    model = get_model(config_path, device=device)
    rm = model.model2

    rm.L = rm.L * fista_lr_mult
    rm.print_every = FISTA_ITERATIONS_BETWEEN_PRINT
    rm.plot = FISTA_PLOT_WITH_PRINT
    rm.iters = FISTA_ITERS
    print(
        f"FISTA params: prior={rm.prox_method}, L={rm.L}, tau={rm.tau}, tv_lambda="
        f"{rm.tv_lambda}, tv_lambdaw={rm.tv_lambdaw}, tv_lambdax={rm.tv_lambdax}"
    )
    rm.iters
    return model


def run_fista_single_image(model, image):
    """ " Run the hyperspectral image through our simulation and reconstruction models."""
    forward_model, recon_model = model.model1, model.model2
    simulated_measurement = forward_model(image.to(recon_model.device))
    recon_model(simulated_measurement.squeeze(dim=(0, 2)).to(recon_model.device))
    return recon_model.out_img


def pred_single_simulation_sample_fista(
    model: SSLSimulationModel, sample_path: str, crop_shape: tuple, patch_shape: tuple
) -> np.ndarray:
    assert os.path.exists(sample_path), f"sample path does not exist {sample_path}"
    patch = scipy.io.loadmat(sample_path)

    gt = patch_predict_utils.prep_image(patch["image"], crop_shape, patch_shape)
    prediction = run_fista_single_image(model=model, image=gt)

    gt_formatted = np.transpose(np.squeeze(gt), (1, 2, 0))
    return prediction, gt_formatted


def pred_single_simulation_sample_learned(
    model: SSLSimulationModel, sample_path: str, crop_shape: tuple, patch_shape: tuple
) -> np.ndarray:
    """
    Run prediction on a single model for a given sample.
    """
    assert os.path.exists(sample_path), f"sample path does not exist {sample_path}"
    patch = scipy.io.loadmat(sample_path)

    gt = patch_predict_utils.prep_image(patch["image"], crop_shape, patch_shape)
    prediction = patch_predict_utils.patchwise_predict_image_learned(
        model=model, image=gt
    )
    gt_formatted = np.transpose(np.squeeze(gt), (1, 2, 0))

    return prediction, gt_formatted


def predict_simulation_samples(
    models: list[SSLSimulationModel], samples: list[tuple], out_dir: str
):
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for sample_name, _, _ in tqdm.tqdm(samples, "Running simulation prediction..."):
        out[sample_name] = {}
        for model in models:
            model_type = "fista"
            if model.model2.__class__.__name__ == "Unet":
                model_type = "learned"

            if model_type == "learned":
                pred, gt = pred_single_simulation_sample_learned(
                    model,
                    os.path.join(SIM_GT_DIRECTORY, sample_name),
                    SIM_CROP_SIZE,
                    SIM_PATCH_SIZE,
                )
            else:
                pred, gt = pred_single_simulation_sample_fista(
                    model,
                    os.path.join(SIM_GT_DIRECTORY, sample_name),
                    SIM_CROP_SIZE,
                    SIM_PATCH_SIZE,
                )

            pred_file = os.path.join(
                out_dir, os.path.splitext(sample_name)[0] + f"_{model_type}.npy"
            )
            np.save(pred_file, pred)
            out[sample_name][model_type] = pred_file

        gt_file = os.path.join(out_dir, os.path.splitext(sample_name)[0] + ".npy")
        np.save(gt_file, gt)
        out[sample_name]["truth"] = gt_file

    with open(os.path.join(out_dir, "sim_preds_directory.json"), "w") as f:
        json.dump(out, f, indent=4, sort_keys=True)
    return out


def main():
    # Fista can recosntruct at full resolution, but the simulation-ready learned model
    # was trained on a small patch, so it needs patchwise prediction & stitching
    learned_smallpatch_config = helper.read_config(LEARNED_CONFIG_PATH)
    fista_bigpatch_config = helper.read_config(FISTA_CONFIG_PATH)

    learned_model = get_model(learned_smallpatch_config, device="cuda:2")
    print(learned_model.model1.operations)

    fista_model = get_fista_model(fista_bigpatch_config, device="cuda:2")
    print(fista_model.model1.operations)

    # For these visualizations, we are going to remove the presence of noise.
    learned_model.model1.operations["fwd_mask_noise"] = False
    learned_model.model1.operations["shot_noise"] = False
    learned_model.model1.operations["read_noise"] = False
    fista_model.model1.operations["fwd_mask_noise"] = False
    fista_model.model1.operations["shot_noise"] = False
    fista_model.model1.operations["read_noise"] = False

    simulation_sample_registry = predict_simulation_samples(
        models=[learned_model, fista_model], samples=SIM_SAMPLES, out_dir=OUT_DIR
    )

    # Then we will increase the presence of noise progressively for a select sample
    noise_sample_demonstration_index = 6

    learned_model.model1.operations["fwd_mask_noise"] = True
    learned_model.model1.operations["shot_noise"] = True
    learned_model.model1.operations["read_noise"] = True
    fista_model.model1.operations["fwd_mask_noise"] = True
    fista_model.model1.operations["shot_noise"] = True
    fista_model.model1.operations["read_noise"] = True

    # Start with low noise. These are the highest noise settings before significant
    # quality dropoff according to study in our paper
    low_read_noise = 1e-3
    low_shot_noise = 1e3
    low_calibration_noise = 1e-3
    learned_model.model1.read_noise_intensity = low_read_noise
    learned_model.model1.shot_noise_photon_count = low_shot_noise
    learned_model.model1.mask_noise_intensity = low_calibration_noise
    fista_model.model1.read_noise_intensity = low_read_noise
    fista_model.model1.shot_noise_photon_count = low_shot_noise
    fista_model.model1.mask_noise_intensity = low_calibration_noise

    low_noise_sample_registry = predict_simulation_samples(
        models=[learned_model, fista_model],
        samples=SIM_SAMPLES[
            noise_sample_demonstration_index : noise_sample_demonstration_index + 1
        ],
        out_dir=os.path.join(OUT_DIR, "low_noise"),
    )

    # Move to a noise value where fista demonstrated considerable decline, but the
    # learned model does not
    medium_read_noise = 1e-2
    medium_shot_noise = 5e3
    medium_calibration_noise = 1e-2
    learned_model.model1.read_noise_intensity = medium_read_noise
    learned_model.model1.shot_noise_photon_count = medium_shot_noise
    learned_model.model1.mask_noise_intensity = medium_calibration_noise
    fista_model.model1.read_noise_intensity = medium_read_noise
    fista_model.model1.shot_noise_photon_count = medium_shot_noise
    fista_model.model1.mask_noise_intensity = medium_calibration_noise
    fista_model.model2.tv_lambda = (
        fista_model.model2.tv_lambda * 10
    )  # Scale tv to deal with the noise
    fista_model.model2.iters = 150  # converges faster with noise

    medium_noise_sample_registry = predict_simulation_samples(
        models=[learned_model, fista_model],
        samples=SIM_SAMPLES[
            noise_sample_demonstration_index : noise_sample_demonstration_index + 1
        ],
        out_dir=os.path.join(OUT_DIR, "medium_noise"),
    )

    # Move to an extremely high noise value, where both models inevitably fail
    high_read_noise = 5e-2
    high_shot_noise = 1e2
    high_calibration_noise = 5e-2
    learned_model.model1.read_noise_intensity = high_read_noise
    learned_model.model1.shot_noise_photon_count = high_shot_noise
    learned_model.model1.mask_noise_intensity = high_calibration_noise
    fista_model.model1.read_noise_intensity = high_read_noise
    fista_model.model1.shot_noise_photon_count = high_shot_noise
    fista_model.model1.mask_noise_intensity = high_calibration_noise
    fista_model.model2.tv_lambda = (
        fista_model.model2.tv_lambda * 20
    )  # Scale tv again to deal with the noise
    fista_model.model2.iters = 150  # converges faster with noise

    high_noise_sample_registry = predict_simulation_samples(
        models=[learned_model, fista_model],
        samples=SIM_SAMPLES[
            noise_sample_demonstration_index : noise_sample_demonstration_index + 1
        ],
        out_dir=os.path.join(OUT_DIR, "high_noise"),
    )

    all_results = (
        simulation_sample_registry
        + low_noise_sample_registry
        + medium_noise_sample_registry
        + high_noise_sample_registry
    )
    print(f"Results: \n\t {all_results}")
    return all_results


if __name__ == "__main__":
    FISTA_PLOT_WITH_PRINT = False
    main()
