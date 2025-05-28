# SelfGNN: Graph Neural Network for Molecular Property Prediction

A machine learning framework for molecular property prediction using Graph Neural Networks (GNNs). This repository implements various graph neural network architectures, specifically designed for processing molecular graphs derived from SMILES representations.

## Project Overview

SelfGNN leverages PyTorch Lightning and PyTorch Geometric to build robust and scalable graph neural network models for predicting molecular properties such as melting point, density, viscosity, and speed.

### Key Components

- **Graph Neural Networks**: Implementation of Graph Attention Networks (GAT) and other graph neural architectures
- **Cross-validation**: Support for k-fold cross-validation to ensure model robustness
- **Experiment Tracking**: Integration with MLflow for experiment tracking and visualization
- **Hyperparameter Optimization**: Using Optuna for hyperparameter tuning
- **Molecular Processing**: Utilities for processing SMILES strings into graph representations

## Project Structure

- `src/`: Source code for the project
  - `model.py`: GNN model implementations (GAT, etc.)
  - `module.py`: Lightning module for training and validation
  - `datamodule.py`: Data handling with PyTorch Lightning
  - `dataset.py`: Dataset implementation for molecular graphs
  - `main.py`: Entry point for training and evaluation
  - `config/`: Configuration files for different experiments
  - `utils/`: Utility functions for graph processing and augmentation

- `data/`: Directory for raw and processed molecular data

## Usage

To train a model with default configuration:

```bash
python -m src.main --experiment default --run_name my_experiment
```

To run k-fold cross-validation:

```bash
python -m src.main --experiment mp_base --run_name mp_cv --run_nfolds 10
```