from math import sqrt
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
import scipy.linalg as scilin
from time import time

from inversion import *

# YOU CAN ADAPT THESE PARAMETERS
use_ray = True
snr = 10.
alpha0 = 1
tau = 1.2
q = 0.8
j = 5
h = 1e-3
scaling_factor = 0.25

# GENERATE DATA
true_image = shepp_logan_phantom()
true_image = rescale(true_image, scale=scaling_factor, mode='reflect', multichannel=False)
n1, n2 = true_image.shape
theta = np.linspace(0., 180., max(n1, n2), endpoint=False)
y_true = radon(true_image, theta=theta, circle=False).flatten()
m = y_true.size
sigma = np.linalg.norm(y_true) / (snr * sqrt(m))
noise = sigma * np.random.randn(m)
y = y_true + noise
# rescale everything
scale = sigma * sqrt(m)
y = y / scale
y_true = y_true / scale
def fwd(u):
    u_img = np.reshape(u, (n1, n2))
    y_img = radon(u_img, theta=theta, circle=False)
    return y_img.flatten() / scale

delta = np.linalg.norm(y - y_true)
print(f"||y - y_true|| = {delta:6f}")

n = n1 * n2

print(f"Image size: {n1}x{n2}")
print(f"Parameter dimension: n={n}")
print(f"Measurement dimension: m={m}")
print(f"Initial ensemble size: j={j}")
prior_mean = np.zeros(n)


c = cov_loc(n, h)

options = {"parallel": use_ray, "alpha": alpha0, "delta": delta, "tau": tau}

# NEXT, COMPUTE ENSEMBLE TIKHONOV ESTIMATES, FIRST WITH NAIVE, THEN WITH TRIDIAGONAL SAMPLING
options["j"] = j
options["sampling"] = "ensemble"
options["c"] = sqrt(q)
t0 = time()
x_eki, traj_eki = solve("adaptive_eki", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)
t_eki = time() - t0

options["c"] = q

options["sampling"] = "nystroem"
t0 = time()
x_nys, traj_nys = solve("adaptive_eki", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)
t_nys = time() - t0

options["sampling"] = "svd"
t0 = time()
x_svd, traj_svd = solve("adaptive_eki", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)
t_svd = time() - t0

# COMPARE RECONSTRUCTION ERROR AND COMPUTATION TIME
x_true = true_image.flatten()
normalization = np.linalg.norm(x_true)

def e_rel(x):
    return np.linalg.norm(x - x_true) / normalization

# PLOT THE RECONSTRUCTION ERROR WRT ITERATION
traj_eki = np.array(traj_eki)
traj_nys = np.array(traj_nys)
traj_svd = np.array(traj_svd)
e_eki = np.linalg.norm(traj_eki - x_true, axis=1) / normalization
e_nys = np.linalg.norm(traj_nys - x_true, axis=1) / normalization
e_svd = np.linalg.norm(traj_svd - x_true, axis=1) / normalization

print(f"EKI computation time: {t_eki}")
print(f"nys-EKI computation time: {t_nys}")
print(f"svd-EKI computation time: {t_svd}")

# PLOTTING
plt.plot(np.arange(e_eki.size), e_eki, 'ro--', label="EKI")
plt.plot(np.arange(e_nys.size), e_nys, 'bx--', label="Nyström-EKI")
plt.plot(np.arange(e_svd.size), e_svd, 'gv--', label="SVD-EKI")
plt.legend(loc="upper right")
plt.xlabel('k')
plt.ylabel(r"$e_\mathrm{rel}$")
plt.savefig("out/speedplot.png")