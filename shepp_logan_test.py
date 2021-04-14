from math import sqrt
import numpy as np
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
j = 2000
tau = 1.2
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


c = cov_loc(n, h, kernel="exponential")

# START WITH DETERMINISTIC TIKHONOV, ALSO TO DETERMINE ALPHA(DELTA)
options = {"parallel": use_ray, "alpha": alpha0, "delta": delta, "tau": tau}
x_tik, alpha_delta = solve("iterative_tikhonov", "deterministic", fwd, y, mean=prior_mean, options=options)

options["alpha"] = alpha_delta
options["j"] = j

# NEXT, COMPUTE ENSEMBLE TIKHONOV ESTIMATES, FIRST WITH NAIVE, THEN WITH TRIDIAGONAL SAMPLING
options["sampling"] = "ensemble"
x_eki = solve("tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)

options["sampling"] = "nystroem"
x_nys = solve("tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)

options["sampling"] = "svd"
x_svd = solve("tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)

# COMPARE RECONSTRUCTION ERROR AND COMPUTATION TIME
x_true = true_image.flatten()
normalization = np.linalg.norm(x_true)

def e_rel(x):
    return np.linalg.norm(x - x_true) / normalization

e_tik = e_rel(x_tik)
e_eki = e_rel(x_eki)
e_nys = e_rel(x_nys)
e_svd = e_rel(x_svd)

print(f"alpha_delta: {alpha_delta}")

print(f"Tikhonov error: {e_tik}")
print(f"EKI error: {e_eki}")
print(f"Nyström-EKI error: {e_nys}")
print(f"SVD-EKI error: {e_svd}")

# COMPARE THE RECONSTRUCTIONS VISUALLY
im_tik = np.reshape(x_tik, (n1, n2))
im_eki = np.reshape(x_eki, (n1, n2))
im_nys = np.reshape(x_nys, (n1, n2))
im_svd = np.reshape(x_svd, (n1, n2))

plt.imshow(true_image, cmap="gray")
plt.savefig("out/shepp_logan_original.png", bbox_inches='tight')
plt.imshow(im_tik, cmap="gray")
plt.savefig("out/shepp_logan_tik.png", bbox_inches='tight')
plt.imshow(im_eki, cmap="gray")
plt.savefig("out/shepp_logan_eki.png", bbox_inches='tight')
plt.imshow(im_nys, cmap="gray")
plt.savefig("out/shepp_logan_nys.png", bbox_inches='tight')
plt.imshow(im_svd, cmap="gray")
plt.savefig("out/shepp_logan_svd.png", bbox_inches='tight')
