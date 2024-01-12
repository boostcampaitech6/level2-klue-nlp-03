#!/bin/sh
# If conda has already installed
conda config --add channels conda-forge
conda update -n base -c defaults conda
conda install -c conda-forge mamba
mamba env create -f environment.yaml

## If env name is 'klue'
# conda init zsh
# conda activate klue
