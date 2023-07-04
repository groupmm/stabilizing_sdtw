# Stabilizing Training with Soft Dynamic Time Warping: A Case Study for Pitch Class Estimation with Weakly Aligned Targets
Code base for reproducing the results presented in 

> J. Zeitler, S. Deniffel, M. Krause, and M. Müller.
> "Stabilizing Training with Soft Dynamic Time Warping: A Case Study for Pitch Class Estimation with Weakly Aligned Targets",
> in Proc. of the 24th Int. Society for Music Information Retrieval Conf., Milan, Italy, 2023.

&copy; Johannes Zeitler (johannes.zeitler@audiolabs-erlangen.de), 2023



## Overview
- **./Code/**: contains all code for running the experiments

- **./Data/**: Location for Schubert Winterreise Dataset and computed HCQTs

- **./Logs/**: Training log files

- **./Models/**: All trained models from the paper

## Getting started
- download the Schubert Winterreise Dataset (SWD) from [Zenodo](https://zenodo.org/record/3968389) and store it in ./Data/
- install the conda environment (conda env create -f environment.yml)
- run ./Code/01_Preprocessing.ipynb to compute HCQTs for the SWD recordings
- run ./Code/02_Training.ipynb to train deep chroma estimators with SDTW loss
- run ./Code/03_Evaluation.ipynb to compute F-measures of trained models (= reproduce the results from Table 2 in the paper)
- run ./Code/04_SoftAlignmentMatrix.ipynb to compute the soft alignment matrices (= reproduce Figure 4 in the paper)

## Acknowledgements
This work was supported by the German Research Foundation (DFG MU 2686/7-2). The authors are with the International Audio Laboratories Erlangen, a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.

This repository builts upon code from [christofw/multipitch_mctc](https://github.com/christofw/multipitch_mctc) and [Maghoumi/pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda).

