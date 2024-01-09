#!/bin/sh
# If conda has already installed
conda config --add channels conda-forge
conda update -c defaults conda
conda install -c conda-forge mamba
mamba env create -f environment.yaml
