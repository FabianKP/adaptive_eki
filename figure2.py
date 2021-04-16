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

from inversion import *
from inversion.simulate_measurement import simulate_measurement

# YOU CAN ADAPT THESE PARAMETERS
use_ray = True
snr = 10.
alpha0 = 1
tau = 1.2
q = 0.8
j1 = 5
h = 1e-3
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
c0 = ornstein_uhlenbeck(n, h)

options = {"parallel": use_ray, "alpha": alpha0, "delta": delta, "tau": tau}

## PERFORM NUMERICAL EXPERIMENT

# apply adaptive Standard-EKI
options["j"] = j1
options["sampling"] = "standard"
options["c0"] = sqrt(q)
x_std, traj_std = adaptive_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, delta=delta, options=options)
# apply adaptive Nyström-EKI
options["c0"] = q
options["sampling"] = "nystroem"
x_nys, traj_nys = adaptive_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, delta=delta, options=options)
# apply adaptive SVD-EKI
options["sampling"] = "svd"
x_svd, traj_svd = adaptive_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, delta=delta, options=options)


def e_rel(x_hat):
    return np.linalg.norm(x_hat - x, axis=1) / np.linalg.norm(x)

# plot the reconstruction error with respect to the iteration number
traj_std = np.array(traj_std)
traj_nys = np.array(traj_nys)
traj_svd = np.array(traj_svd)
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