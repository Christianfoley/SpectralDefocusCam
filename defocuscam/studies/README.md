## Studies
This directory contains code for reproducing the studies performed in our paper. Each subdirectory contains the necessary code to generate results for a particular simulation or experimental result presented in the paper, as well as a set of `configs` for configuring the models and codebase according to the parameters of each element in the study. 

Where relevant, each study directories also contains a `baseline_results` with visualizations against which you can compare the results you generate.

### Experimental results
This directory contains a [notebook](experimental_results/generate_select_recons.ipynb) for reproducing our experimental reconstructions from the raw camera measurement and calibration data and our pretrained models.

### Simulation Methods Comparison
This directory contains a [notebook](simulation_methods_comparison_collage/generate_select_recons.ipynb) for reproducing simulation reconstructions with our method and competing methods using our pretrained models.

### Simulation Methods Comparison Table
This directory contains a script for running dataset-wide benchmarks for our and competing methods on simulated data. Run the following command from the root directory:

```
conda activate defocuscam \
python defocuscam/studies/simulation_methods_comparison_table/run.py
```

NOTE: you will be prompted to login to wandb.

### Simulation Methods Comparison Table
This directory contains a script for running dataset-wide benchmarks on our method's tolerance to noise. Run the following command from the root directory:

```
conda activate defocuscam \
python defocuscam/studies/simulation_noise_tolerance_ablation/run.py
```

NOTE: you will be prompted to login to wandb.

### Simulation Methods Comparison Table
This directory contains a script for running dataset-wide benchmarks to study the impact of the number of measurements taken at multiple levels of focus on our system's reconstruction quality. Run the following command from the root directory:

```
conda activate defocuscam \
python defocuscam/studies/simulation_numbers_of_defocus_ablation/run.py
```

NOTE: you will be prompted to login to wandb.

### Simulation Resolution Study
This directory contains a script for running simulation benchmarks to study our system's theoretical two-point resolution. First, generate the two-point resolution volumes by running the [resolution_volume_generation.ipynb](simulation_resolution_study/resolution_volume_generation.ipynb) notebook.

Then, run the following command from the root directory:

```
conda activate defocuscam \
python defocuscam/studies/simulation_resolution_study/run.py
```

NOTE: you will be prompted to login to wandb.