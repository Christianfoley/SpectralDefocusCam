import studies.run_study_utils as run_study_utils

import pathlib
import os
import glob

# We run the same configs for each model, but overriding with
# all of these parameters
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
    10,
    50,
    100,
    500,
    1000,
    10000,
    100000,
    500000,
    1000000,
]


def main():
    configs = glob.glob(os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml"))

    sweep_config_paths = []
    sweep_config_overrides = []
    sweep_name_suffixes = []
    for config_path in configs:
        for sigma, photon_count in zip(READ_NOISE_INTENSITY, SHOT_NOISE_PHOTON_COUNT):
            sweep_config_overrides.append(
                {
                    "forward_model_params.mask_noise.intensity": float(sigma),
                    "forward_model_params.sample_noise.intensity": float(sigma),
                    "forward_model_params.sample_noise.photon_count": float(
                        photon_count
                    ),
                }
            )
            sweep_config_paths.append(config_path)
            sweep_name_suffixes.append(f"read={sigma}_shot={photon_count}")

    run_study_utils.run_study_sweep(
        study_name="simulation_noise_tolerance_ablation",
        config_paths=sweep_config_paths,
        overrides=sweep_config_overrides,
        study_name_suffixes=sweep_name_suffixes,
        overwrite_existing_recons=False,
        overwrite_existing_metrics=False,
    )


if __name__ == "__main__":
    main()
