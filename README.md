# Dewan Lab PID Analysis

This repository contains the python package required to read, repackage, and graph data collected using the Dewan Lab's PID setup. This package currently only supports the 'new' Bpod-based PID setup files, but will eventually be updated to be backwards compatible with the older Arduino-based PID setup H5 files.


# Installation
> Note: It is recommended to use the included `environment.yaml` file to install the package. However, the package can also be installed manually if desired.
	

## Anaconda / Conda / Mamba

Ensure you have a working installation of Anaconda. We recommend using mamba which is included in [miniforge](https://github.com/conda-forge/miniforge).
1. Download the `environment.yaml` file from this repository
2. Run the command `mamba env create -f [PATH_TO_FILE]/environment.yaml` to automatically install the package and all required dependencies

## Manually
1. Clone the repository and navigate into the directory
2. Run `pip install .` within the cloned repository to install the package in the currently active python environment

> Alternative Install: You can run `pip install git+https://github.com/OlfactoryBehaviorLab/PID_Analysis.git` to download the package directly from GitHub without the need to clone it.

# Using the Package
> Note: If using Anaconda/Conda/Mamba, be sure to activate the PID environment by running `conda activate PID`

To run the package, simply open a terminal and run `python -m pid_analysis` and select the *.mat* file(s) to process.
