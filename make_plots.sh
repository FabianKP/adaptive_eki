#!/bin/bash

# Generates all plots for the .
# Needs to be executed in the numerics-directory.

python3 shepp_logan_test.py
python3 speedplot.py
python3 convergence_wrt_ensemblesize.py
python3 divergence_wrt_alpha.py

