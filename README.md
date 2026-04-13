# IFLS: Integrated Federated Loss Shapley

Code accompanying the dissertation *"Integrated Federated Loss Shapley:
Client Contribution Evaluation for
Tree-Based Ensemble Models in
Federated Learning"*.

## Overview

This repository contains the implementation of IFLS and all baseline methods (Exact Shapley, GTG-Shapley, Permutation Shapley), along with the experimental pipeline used to produce the dissertation results.

## Repository Structure

- `experiment_utils.py` — core implementations: IFLS, baselines, federated ensemble, data partitioning, and evaluation metrics
- `Experiments.ipynb` — reproduces all experimental results across K=5, 10, 15 clients under Dirichlet-skewed partitioning and experiment for IID label skew with different data sizes across clients
- `plot_results.ipynb` — generates all figures from saved results

## Requirements

Python 3.x. Install dependencies with:
```bash
pip install -r requirements.txt
```

## Reproducing Results

1. Run `Experiments.ipynb` top to bottom — this will download the Adult and Bank Marketing datasets automatically via OpenML and save results
2. Run `plot_results.ipynb` to generate figures from saved results

## Dataset
Experiments use two datasets fetched automatically via `sklearn.datasets.fetch_openml`:

- [Adult Income dataset](https://www.openml.org/d/1590) from OpenML
- [Bank Marketing dataset](https://www.openml.org/d/1461) from OpenML
