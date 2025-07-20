import studies.run_study_utils as run_study_utils
import pathlib
import os
import glob


def main():
    configs = glob.glob(os.path.join(pathlib.Path(__file__).parent, "configs", "*.yml"))

    run_study_utils.run_study_sweep(
        study_name="simulation_resolution_study",
        config_paths=configs,
        overwrite_existing_recons=False,
        overwrite_existing_metrics=False,
    )


if __name__ == "__main__":
    main()
