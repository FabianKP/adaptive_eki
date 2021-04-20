"""
Executing this file reproduces figure 5 from the paper.
For fixed regularization parameter alpha, the error between
Tikhonov regularization and Standard-, Nyström-, and SVD-EKI is plotted for varying sample size J
"""


import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt
import numpy as np

from inversion import *


# YOU CAN ADAPT THESE PARAMETERS
use_ray = True  # toggles parallelization; set False for final experiment.
alpha = 0.03
snr = 10. # desired signal-to-noise ratio
scaling_factor = 0.25 # determines the dimension; set to 0.4 for final experiment
h = 0.01

# obtain data
y_im, y_hat_im, x_im, fwd, delta = simulate_measurement(snr, scaling_factor)
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
x0 = np.zeros(n)
c0 = ornstein_uhlenbeck(n1, n2, h)
print("Computing SVD of c0...")
c0_root, evals, evecs = matrix_sqrt(c0)
print("...done.")

# determine a good value for the regularization parameter alpha using the discrepancy principle
# and Tikhonov regularization
options = {"parallel": use_ray}
x_tik = tikhonov(fwd=fwd, y=y_hat, x0=x0, c0_root=c0_root, alpha=alpha, options=options)

# apply EKI
traj_std = []
traj_nys = []
traj_svd = []
sizes = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000, 8000]
options["c0_root"] = c0_root
options["c0_eigenvalues"] = evals
options["c0_eigenvectors"] = evecs
for j in sizes:
    options["j"] = j
    print("Sample size: ", j)
    options["sampling"] = "standard"
    x_std = direct_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, alpha=alpha, options=options)
    traj_std.append(x_std)
    options["sampling"] = "nystroem"
    x_nys = direct_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, alpha=alpha, options=options)
    traj_nys.append(x_nys)
    options["sampling"] = "svd"
    x_svd = direct_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, alpha=alpha, options=options)
    traj_svd.append(x_svd)

# compute approximation errors
def approximation_error(x_hat):
    return np.linalg.norm(x_hat - x_tik[:,np.newaxis], axis=0) / np.linalg.norm(x)
traj_std = np.array(traj_std).T
traj_nys = np.array(traj_nys).T
traj_svd = np.array(traj_svd).T
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

