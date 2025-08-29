import defocuscam.studies.run_study_utils as run_study_utils
import pathlib
import os
import glob
import torch


def main():
    configs = glob.glob(os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml"))
    overrides = None if torch.cuda.is_available() else [{"device": "cpu"}] * len(configs)

    run_study_utils.run_study_sweep(
        study_name="simulation_numbers_of_defocus_ablation",
        config_paths=configs,
        overwrite_existing_recons=False,
        overwrite_existing_metrics=False,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
