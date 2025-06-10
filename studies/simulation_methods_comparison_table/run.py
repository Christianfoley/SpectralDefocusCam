import studies.run_study_utils as run_study_utils
import pathlib
import os
import glob
import time

import json
import wandb
import utils.helper_functions as helper


def main():
    configs = glob.glob(os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml"))

    print(f"Collected {len(configs)} configs:")
    for config_path in configs:
        print("\t", os.path.abspath(config_path))
    print("\n")

    print("Beginning run")
    start = time.time()

    for config_path in configs:
        wandb_run = wandb.init(
            project="SpectralDefocusCam",
            name="simulation_methods_comparison_table",
            config=helper.read_config(config_path),
        )

        try:
            reconstructed_files = run_study_utils.run_reconstruction_grid(config_path)
            wandb.log({"Num_valid_recons": len(reconstructed_files)})
        except Exception as e:
            print(f"Reconstructions failed: {str(e)}")

        try:
            metrics_path = run_study_utils.compute_metrics(config_path)
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
