# Profit Maximization for a Robotics-as-a-Service Model

This repository implements the methods described in the paper "Profit Maximization for a Robotics-as-a-Service Model". It provides a simulation framework for optimizing pricing and replacement decisions in a RaaS setting using data-driven models for customer behavior and robot degradation.

For details on the methodology, please refer to the [paper PDF](profit_maximization_for_a_raas_model.pdf)

## Installation

### Using Mamba (Recommended)
1. Create and activate the environment from `requirements.yml`:
```
mamba env create -f requirements.yml
mamba activate degradable
```
2. Install the package in editable mode:
`pip install -e .`

### Using Pip
If not using Mamba:
```
pip install -r requirements.yml
pip install -e .
```

## Usage

### Running Simulations
Use the notebook in `notebooks/simulate.ipynb` to run experiments:
- Configure parameters in `src/raas/config.py`.
- Execute cells to simulate and analyze results.

### Plotting Results
Use `notebooks/plotting.ipynb` for visualizations:
- Load simulation outputs and generate plots (e.g., policy thresholds, profit rates).

### Extending the Code
- Core logic is in `src/raas/` (e.g., `simulation.py` for the main simulator).
- Experiments/scripts in `src/experiments/` (add custom runners here).

Example command-line usage (if you add a script):

`python -m experiments.main --skip_training [True/False]`
Setting `skip_training` lets you skip the projected volume algorithm and starts with perfect knowledge of $\hat u = u$.

## Repository Structure
- `src/raas/`: Core package with models, learners, and simulator.
- `src/experiments/`: Scripts for running experiments.
- `notebooks/`: Jupyter notebooks for simulation and plotting.
- `setup.py`: For package installation.
- `requirements.yml`: Virtual environment specs.