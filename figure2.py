"""
Executing this file reproduces figure 2 of the paper.
The adaptive Standard-EKI, adaptive Nyström-EKI and adaptive SVD-EKI methods are compared
against each other on reconstructing the Shepp-Logan phantom from noisy measurements.
"""

from math import sqrt
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import scipy.linalg as scilin

from inversion import *
from inversion.simulate_measurement import simulate_measurement

# YOU CAN ADAPT THESE PARAMETERS
use_ray = True
snr = 10.
alpha0 = 1
tau = 1.2
q = 0.8
j1 = 50
h = 0.01
scaling_factor = 0.25

# obtain data
y_hat_im, x_im, fwd, delta = simulate_measurement(snr, scaling_factor)
n1, n2 = x_im.shape
x = x_im.flatten()
y_hat = y_hat_im.flatten()
n = x.size
m = y_hat.size

# Display information
print(f"Image size: {n1}x{n2}")
print(f"Parameter dimension: n={n}")
print(f"Measurement dimension: m={m}")
print(f"Initial sample size: j1={j1}")

# Set up x0 and c0
x0 = np.zeros(n)
c0 = ornstein_uhlenbeck(n1, n2, h)
print("Computing SVD of c0...")
c0_root, evals, evecs = matrix_sqrt(c0)
print("...done.")

options = {"parallel": use_ray, "alpha": alpha0, "delta": delta, "tau": tau}

## PERFORM NUMERICAL EXPERIMENT

# apply adaptive Standard-EKI
options["j"] = j1
options["sampling"] = "standard"
options["c0_root"] = c0_root
options["c"] = sqrt(q)
traj_std, a1 = adaptive_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, delta=delta, options=options)
# apply adaptive Nyström-EKI
options["c"] = q
options["sampling"] = "nystroem"
traj_nys, a2 = adaptive_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, delta=delta, options=options)
# apply adaptive SVD-EKI
options["sampling"] = "svd"
# svd-sampling requires the svd of c0
options["c0_eigenvalues"] = evals
options["c0_eigenvectors"] = evecs
traj_svd, a3 = adaptive_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, delta=delta, options=options)


def e_rel(x_hat):
    return np.linalg.norm(x_hat - x[:, np.newaxis], axis=0) / np.linalg.norm(x)

# plot the reconstruction error with respect to the iteration number
traj_std = np.array(traj_std).T
traj_nys = np.array(traj_nys).T
traj_svd = np.array(traj_svd).T
e_eki = e_rel(traj_std)
e_nys = e_rel(traj_nys)
e_svd = e_rel(traj_svd)

plt.plot(np.arange(e_eki.size), e_eki, 'ro--', label="EKI")
plt.plot(np.arange(e_nys.size), e_nys, 'bx--', label="Nyström-EKI")
plt.plot(np.arange(e_svd.size), e_svd, 'gv--', label="SVD-EKI")
plt.legend(loc="upper right")
plt.xlabel('k')
plt.ylabel(r"$e_\mathrm{rel}$")
plt.savefig("out/figure2.png", bbox_inches='tight')