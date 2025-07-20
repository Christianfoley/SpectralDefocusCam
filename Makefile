# Makefile for Spectral Defocus Camera project

.PHONY: setup-env-cuda setup-env-macos setup-env-linux help

# Environment names
ENV_NAME = defocuscam
ENV_NAME_CUDA = defocuscam-cuda

CONDA_BASE := $(shell conda info --base)

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "  make setup-env-macos     - environment setup for macOS (CPU-only PyTorch)"
	@echo "  make setup-env-linux     - environment setup for Linux (CPU-only PyTorch)"
	@echo "  make setup-env-cuda      - environment setup with CUDA support (Linux/Windows)"
	@echo ""
	@echo "note: conda or miniconda must be installed"

# Installation commands
make-env:
	@echo "Creating conda environment..."
	@conda create -n $(ENV_NAME) python=3.11 -y || true
	@echo "Done!"

make-env-cuda:
	@echo "Creating conda environment with CUDA support..."
	@conda create -n $(ENV_NAME_CUDA) python=3.11 -y || true
	@echo "Done!"

install:
	@echo "Installing dependencies from..."
	bash -c "source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && pip install -e .[cpu] --find-links https://download.pytorch.org/whl/cpu/torch_stable.html"
	@echo "Done!"

install-cuda:
	@echo "Installing dependencies from..."
	bash -c "source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(ENV_NAME_CUDA) && conda install cudatoolkit=11.6 -c conda-forge -y"
	@echo "Installing CUDA dependencies..."
	bash -c "source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(ENV_NAME_CUDA) && pip install -e .[cuda] --find-links https://download.pytorch.org/whl/cu116/torch_stable.html"
	@echo "Done!"


# Set up environment for macOS (CPU-only PyTorch, no CUDA)
setup-env-macos:
	@echo "Setting up development environment for macOS..."
	$(MAKE) make-env
	$(MAKE) install
	@echo ""
	@echo "✅ Environment setup complete!"
	@echo "        - conda activate $(ENV_NAME)"

# Set up environment for Linux (CPU-only PyTorch, no CUDA)
setup-env-linux:
	@echo "Setting up development environment for Linux (CPU-only)..."
	$(MAKE) make-env
	$(MAKE) install
	@echo ""
	@echo "✅ Environment setup complete!"
	@echo "        - conda activate $(ENV_NAME)"

# Set up environment with CUDA support
setup-env-cuda:
	@echo "Setting up development environment with CUDA support..."
	$(MAKE) make-env-cuda
	$(MAKE) install-cuda
	@echo ""
	@echo "✅ Environment setup complete!"
	@echo "        - conda activate $(ENV_NAME_CUDA)"

