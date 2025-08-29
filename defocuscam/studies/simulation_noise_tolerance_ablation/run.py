import defocuscam.studies.run_study_utils as run_study_utils

import pathlib
import os
import glob
import torch

# We run the same configs for each model, but overriding with
# all of these parameters
CALIBRATION_NOISE_INTENSITY = [
    0.000001,
    0.000005,
    0.00001,
    0.00005,
    0.0001,
    0.001,
    0.01,
    0.05,
    0.1,
]
READ_NOISE_INTENSITY = [
    0.000001,
    0.000005,
    0.00001,
    0.00005,
    0.0001,
    0.001,
    0.01,
    0.05,
    0.1,
]
SHOT_NOISE_PHOTON_COUNT = [
    1000000,
    500000,
    100000,
    10000,
    1000,
    500,
    100,
    50,
    10,
]


def generate_sweep_configs(configs):
    """
    Unlike other studies, here we build the configs in the script, to ensure we're just
    varying a single set of parameters for many values
    """
    sweep_config_paths = []
    sweep_config_overrides = []
    sweep_name_suffixes = []
    for config_path in configs:
        # generate calibration noise configs
        for mask_noise in CALIBRATION_NOISE_INTENSITY:
            sweep_config_overrides.append(
                {
                    "device": 1,
                    "forward_model_params.mask_noise.intensity": float(mask_noise),
                    "forward_model_params.operations.fwd_mask_noise": True,
                }
            )
            sweep_config_paths.append(config_path)
            sweep_name_suffixes.append(f"mask={mask_noise}")

        # generate shot noise configs
        for photons in SHOT_NOISE_PHOTON_COUNT:
            sweep_config_overrides.append(
                {
                    "device": 0,
                    "forward_model_params.sample_noise.photon_count": float(photons),
                    "forward_model_params.operations.shot_noise": True,
                }
            )
            sweep_config_paths.append(config_path)
            sweep_name_suffixes.append(f"shot={photons}")

        # generate read noise configs
        for read_noise in READ_NOISE_INTENSITY:
            sweep_config_overrides.append(
                {
                    "device": 2,
                    "forward_model_params.sample_noise.intensity": float(read_noise),
                    "forward_model_params.operations.read_noise": True,
                }
            )
            sweep_config_paths.append(config_path)
            sweep_name_suffixes.append(f"read={read_noise}")
    return sweep_config_paths, sweep_config_overrides, sweep_name_suffixes


def main():
    configs = glob.glob(os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml"))
    config_paths, overrides, suffixes = generate_sweep_configs(configs)

    if not torch.cuda.is_available():
        for override in overrides:
            override.update({"device": "cpu"})

    run_study_utils.run_study_sweep(
        study_name="simulation_noise_tolerance_ablation",
        config_paths=config_paths,
        overrides=overrides,
        study_name_suffixes=suffixes,
        overwrite_existing_recons=False,
        overwrite_existing_metrics=False,
    )


if __name__ == "__main__":
    main()
