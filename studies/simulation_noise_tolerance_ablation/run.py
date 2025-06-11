import studies.run_study_utils as run_study_utils
import pathlib
import os
import glob
import time

import json
import wandb
import utils.helper_functions as helper

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

    print(f"Collected {len(configs)} configs:")
    for config_path in configs:
        print("\t", os.path.abspath(config_path))
    print("\n")

    print("Beginning run")
    start = time.time()

    for config_path in configs:
        for sigma, photon_count in zip(READ_NOISE_INTENSITY, SHOT_NOISE_PHOTON_COUNT):
            overrides = {
                "forward_model_params.mask_noise.intensity": float(sigma),
                "forward_model_params.sample_noise.intensity": float(sigma),
                "forward_model_params.sample_noise.photon_count": float(photon_count),
            }

            wandb_run = wandb.init(
                project="SpectralDefocusCam",
                name=(
                    f"simulation_noise_tolerance_ablation_{pathlib.Path(config_path).stem}"
                    f"_read={sigma}_shot={photon_count}"
                ),
                config=run_study_utils._override_config_parameters(
                    helper.read_config(config_path), overrides
                ),
            )

            try:
                reconstructed_files = run_study_utils.run_reconstruction_grid(
                    config_path, override_params=overrides
                )
                wandb.log({"Num_valid_recons": len(reconstructed_files)})
            except Exception as e:
                print(f"Reconstructions failed: {str(e)}")

            try:
                metrics_path = run_study_utils.compute_metrics(
                    config_path, override_params=overrides
                )
                with open(metrics_path, "r") as f:
                    wandb.log(json.load(f))
            except Exception as e:
                print(f"Metrics computation failed:{str(e)}")

            wandb_run.finish()

    print("\n")
    print("Done!")
    print(f"Total time: {time.time() - start:.2f} seconds.")


if __name__ == "__main__":
    main()
