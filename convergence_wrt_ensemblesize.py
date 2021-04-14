
from math import sqrt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

from inversion import *


# YOU CAN ADAPT THESE PARAMETERS
use_ray = True  # toggles parallelization; set False for final experiment.
alpha = 0.015
snr = 10. # desired signal-to-noise ratio
scaling_factor = 0.25 # determines the dimension; set to 0.4 for final experiment
h = 1e-3

# GENERATE DATA
true_image = shepp_logan_phantom()
true_image = rescale(true_image, scale=scaling_factor, mode='reflect', multichannel=False)
n1, n2 = true_image.shape
theta = np.linspace(0., 180., max(n1,n2), endpoint=False)
y_true = radon(true_image, theta=theta, circle=False).flatten()

# CREATE NOISY MEASUREMENTS
m = y_true.size
sigma = np.linalg.norm(y_true) / (snr * sqrt(m))
noise = sigma * np.random.randn(m)
y = y_true + noise

# DISPLAY SOME INFOS
n = n1 * n2
print(f"Parameter dimension n_u: {n}")
print(f"Measurement dimension n_y: {m}")
print("Signal-to-noise ratio: ", np.linalg.norm(y_true)/np.linalg.norm(noise))

# NORMALIZE MEASUREMENT AND OBSERVATION OPERATOR
scale = sigma*sqrt(m)
y = y / scale
y_true = y_true / scale
def fwd(u):
    u_img = np.reshape(u, (n1,n2))
    y_img = radon(u_img, theta=theta, circle=False)
    return y_img.flatten() / scale


prior_mean = np.zeros(n)
c = cov_loc(n, h, kernel="exponential")

options = {"alpha": alpha, "parallel": use_ray}

# DETERMINE ALPHA VIA ITERATIVE TIKHONOV
x_tik = solve("tikhonov", "deterministic", fwd, y, mean=prior_mean, cov=c, options=options)

# ENSEMBLE TIKHONOV FOR INCREASING ENSEMBLE SIZE
traj_eki = []
traj_nys = []
traj_svd = []
sizes = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000, 8000]
# you could make this experiment much faster by precomputing the localized covariance matrix once,
# and then reuse it for all computations
for j in sizes:
    options["j"] = j
    print("Ensemblesize: ", j)
    options["sampling"] = "ensemble"
    x_eki = solve("tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)
    traj_eki.append(x_eki)
    options["sampling"] = "nystroem"
    x_nys = solve("tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)
    traj_nys.append(x_nys)
    options["sampling"] = "svd"
    x_svd = solve("tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)
    traj_svd.append(x_svd)

# COMPUTE APPROXIMATION ERRORS
x_true = true_image.flatten()
true_norm = np.linalg.norm(x_true)
reconstruction_error = np.linalg.norm(x_tik - x_true) / true_norm
print(f"Relative reconstruction error {reconstruction_error}")
traj_eki = np.array(traj_eki)
traj_nys = np.array(traj_nys)
traj_svd = np.array(traj_svd)
e_eki = np.linalg.norm(traj_eki - x_tik, axis=1) / true_norm
e_nys = np.linalg.norm(traj_nys - x_tik, axis=1) / true_norm
e_svd = np.linalg.norm(traj_svd - x_tik, axis=1) / true_norm

# PLOTTING
plt.plot(sizes, e_eki, 'ro--', label="EKI")
plt.plot(sizes, e_nys, 'bx--', label="Nyström-EKI")
plt.plot(sizes, e_svd, 'gv--', label="SVD-EKI")
plt.xlabel("J")
plt.ylabel(r"$e_\mathrm{app}$")
plt.legend(loc="upper right")
plt.savefig("out/convergence_wrt_samplesize.png", bbox_inches='tight')

