"""
Executing this file reproduces figure 6 from the paper.
For fixed sample size J, the error between Tikhonov regularization and Standard-, Nyström-, and SVD-EKI is plotted
for varying regularization parameters alpha.
"""


import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scilin

from inversion import *


# YOU CAN ADAPT THESE PARAMETERS
use_ray = True
n_alpha = 30
j = 2000
snr = 10.  # desired signal-to-noise ratio
alpha1 = 1.   # minimal alpha
c = 0.8 # sampling error is computed for alpha, alpha/c0, alpha/c0^2, ...
h = 0.01
scaling_factor = 0.25 # determines the dimension of the problem

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

# create list of alphas
alphas = [alpha1]
alpha = alpha1
for i in range(n_alpha-1):
    alpha = alpha * c
    alphas.append(alpha)

# compute Tikhonov-regularized solutions
options = {"parallel": use_ray}
traj_tik = tikhonov_list(fwd=fwd, y=y_hat, x0=x0, c0_root=c0_root, alphas=alphas, options=options)

# computed direct EKI solutions
options["j"] = j

options["sampling"] = "standard"
options["c0_root"] = c0_root
print("Standard-EKI")
traj_std  = eki_list(fwd=fwd, y=y_hat, x0=x0, c0=c0, alphas=alphas, options=options)
options["sampling"] = "nystroem"
print("Nystroem-EKI")
traj_nys = eki_list(fwd=fwd, y=y_hat, x0=x0, c0=c0, alphas=alphas, options=options)
options["sampling"] = "svd"
options["c0_eigenvalues"] = evals
options["c0_eigenvectors"] = evecs
print("SVD-EKI")
traj_svd = eki_list(fwd=fwd, y=y_hat, x0=x0, c0=c0, alphas=alphas, options=options)

# compute approximation errors
traj_tik = np.array(traj_tik).T
traj_std = np.array(traj_std).T
traj_nys = np.array(traj_nys).T
traj_svd = np.array(traj_svd).T
def approximation_error(traj_hat):
    return np.linalg.norm(traj_hat - traj_tik, axis=0) / np.linalg.norm(x)
e_std = approximation_error(traj_std)
e_nys = approximation_error(traj_nys)
e_svd = approximation_error(traj_svd)

# plotting
plt.plot(alphas, e_std, 'ro--', label="Standard-EKI")
plt.plot(alphas, e_nys, 'bx--', label="Nyström-EKI")
plt.plot(alphas, e_svd, 'gv--', label="SVD-EKI")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$e_\mathrm{app}$")
plt.legend(loc="upper right")
plt.savefig("out/figure6.png", bbox_inches='tight')
