"""
Executing this file reproduces figure 5 from the paper.
For fixed regularization parameter alpha, the error between
Tikhonov regularization and Standard-, Nyström-, and SVD-EKI is plottedfor varying sample size J
"""


import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt
import numpy as np

from inversion import *


# YOU CAN ADAPT THESE PARAMETERS
use_ray = True  # toggles parallelization; set False for final experiment.
alpha = 0.015
snr = 10. # desired signal-to-noise ratio
scaling_factor = 0.25 # determines the dimension; set to 0.4 for final experiment
h = 1e-3

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

# Set up x0 and c0
x_bar = np.zeros(n)
c0 = ornstein_uhlenbeck(n, h)

# determine a good value for the regularization parameter alpha using the discrepancy principle
# and Tikhonov regularization
options = {"alpha": alpha, "parallel": use_ray}
x_tik = solve("tikhonov", "deterministic", fwd, y_hat, mean=x_bar, cov=c0, options=options)

# apply EKI
traj_std = []
traj_nys = []
traj_svd = []
sizes = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000, 8000]
for j in sizes:
    options["j"] = j
    print("Sample size: ", j)
    options["sampling"] = "ensemble"
    x_std = solve("tikhonov", "ensemble", fwd, y_hat, mean=x_bar, cov=c0, options=options)
    traj_std.append(x_std)
    options["sampling"] = "nystroem"
    x_nys = solve("tikhonov", "ensemble", fwd, y_hat, mean=x_bar, cov=c0, options=options)
    traj_nys.append(x_nys)
    options["sampling"] = "svd"
    x_svd = solve("tikhonov", "ensemble", fwd, y_hat, mean=x_bar, cov=c0, options=options)
    traj_svd.append(x_svd)

# compute approximation errors
def approximation_error(x_hat):
    return np.linalg.norm(x_hat - x_tik, axis=1) / np.linalg.norm(x)
traj_std = np.array(traj_std)
traj_nys = np.array(traj_nys)
traj_svd = np.array(traj_svd)
e_eki = approximation_error(traj_std)
e_nys = approximation_error(traj_nys)
e_svd = approximation_error(traj_svd)

# plotting
plt.plot(sizes, e_eki, 'ro--', label="Standard-EKI")
plt.plot(sizes, e_nys, 'bx--', label="Nyström-EKI")
plt.plot(sizes, e_svd, 'gv--', label="SVD-EKI")
plt.xlabel("J")
plt.ylabel(r"$e_\mathrm{app}$")
plt.legend(loc="upper right")
plt.savefig("out/figure5.png", bbox_inches='tight')

