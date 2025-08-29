### 🚧 Under Construction 🚧 ###
Official implementation of *Spectral DefocusCam: super-resolved hyperspectral imaging from defocus*, presented at ICCP 2025
Previous iterations presented at [COSI](https://opg.optica.org/abstract.cfm?uri=COSI-2022-CF2C.1)

### Environment setup
For mac-os systems:
```
make setup-env-macos
```

For linux systems:
```
make setup-env-linux
```

If you have a cuda-capable gpu (linux systems only):
```
make setup-env-cuda
```

### Data Retrieval
All data for reproducing our results is stored in a public [google drive folder](https://drive.google.com/drive/folders/176sErdN4R-5LPUs3SD2AGebDJkh3lTty). To access this data, you can download it directly from the above google drive link and place each folder underneath the `data/` directory. This is equivalent to running:
```
make data
```

Since this can be slow, you can download individual partitions directly by running following commands in the project root directory, after activating your environment:

**Trained models:**
```
make data-models
```

**Calibration data:** from our experimental prototype.
```
make data-calibration
```

**Experimental measurements:** captured on indoor and outdoor scenes using our experimental prototype.

```
make data-experimental-measurements
```

**Simulation data:** For convenience we make directly available a subset of the simulation data used for training and evaluation of our reconstruction models, from the [Harvard hyperspectral fruit dataset](http://vision.seas.harvard.edu/hyperspec/).

```
make data-simulation
```

### Quick start - reproducing experimental results
To reproduce any of the existing studies, first run `make data` or access and download the data from google drive with your browser, then navigate to [generate_select_recons.ipynb](defocuscam/studies/experimental_results/generate_select_recons.ipynb) and run the notebook using the `defocuscam` environment.

### Reproducing simulation studies
See the [studies/README.md](defocuscam/studies/README.md)