#!/bin/bash

# Create Python environment
conda create --yes --name mltosca python=3

# Setting source
source ~/anaconda3/etc/profile.d/conda.sh

# Activate
conda activate mltosca

# Install packages
conda install -y scikit-learn
