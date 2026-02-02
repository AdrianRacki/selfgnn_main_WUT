# Wide and Cross GNNs: Cross Interactions and Parallel Scaling for Robust Chemical Property Prediction

A Graph Neural Network (GNN) framework for molecular property prediction using PyTorch Lightning and Hydra for configuration management.

## Overview

This repository implements a GNN-based model for predicting various molecular properties from chemical structures. It supports both classification and regression tasks on multiple molecular datasets including:

- **Classification**: BACE, BBBP, HIV, ClinTox
- **Regression**: ESOL, FreeSolv, Lipophilicity (LIPO), Density, Heat Capacity, Melting Point (MP), Speed of Sound, Viscosity

## Installation

This project uses Poetry for dependency management. Install dependencies with:

```bash
poetry install
```

Or activate the environment:

```bash
poetry shell
```

## Training a Model

The main training script is `src/main.py`. It uses Hydra for configuration management, with config files located in `src/config/`.

### Basic Usage

Train a model using a specific experiment configuration:

```bash
python src/main.py --experiment <config_name> --run_name <your_run_name>
```

### Examples

Train on the LIPO dataset:
```bash
python src/main.py --experiment lipo_base --run_name lipo_experiment_1
```

Train on the BACE dataset:
```bash
python src/main.py --experiment bace_base --run_name bace_experiment_1
```

### K-Fold Cross-Validation

Run k-fold cross-validation by specifying the number of folds:

```bash
python src/main.py --experiment lipo_base --run_name lipo_cv --run_nfolds 5
```

### Command Line Arguments

- `--experiment`: Name of the experiment config file (without .yaml extension)
- `--run_name`: Name for this training run (default: "default_run")
- `--run_nfolds`: Number of folds for cross-validation (default: 0, no CV)
- `--predict`: Whether to run predictions after training (default: False)

## Configuration

Experiment configurations are located in `src/config/`. The main config structure includes:

- **data/**: Dataset and datamodule configurations
- **model/**: Model architecture configurations
- **trainer/**: PyTorch Lightning trainer settings
- **callbacks/**: Training callbacks (checkpointing, logging, etc.)
- **metrics/**: Evaluation metrics

Base experiment configs available:
- `bace_base.yaml`, `bbbp_base.yaml`, `hiv_base.yaml`, `tox_base.yaml` (classification)
- `lipo_base.yaml`, `esolv_base.yaml`, `freesolv_base.yaml`, `dens_base.yaml`, etc. (regression)

## Project Structure

```
├── src/
│   ├── main.py              # Main training script
│   ├── config/              # Hydra configuration files
│   ├── model.py             # Model architectures
│   ├── module.py            # Lightning module
│   ├── datamodule.py        # Data handling
│   └── dataset.py           # Dataset classes
├── data/                    # Dataset storage
├── results/                 # Training results
├── notebooks/               # Analysis notebooks
└── mlruns/                  # MLflow experiment tracking
```
