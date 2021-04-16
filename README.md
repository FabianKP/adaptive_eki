# adaptive_eki
This repository contains code accompanying the paper "On convergence rates of ensemble Kalman inversion for linear ill-posed problems"

You can reproduce all figures at once by executing the script "make_plots.sh". 
Depending on your computer, this might take considerable time.
The figures will not look exactly the same since the algorithms are stochastic. 
Also note that the implementation is not optimized for speed. In particular, the
implementation is not suited for comparing the wallclock times of the different methods.

# Requirements #

This code was tested with Python 3.8. It uses the following packages:
- numpy
- scipy
- scikit-image
- matplotlib
- ray

