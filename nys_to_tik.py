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
prior_mean = np.zeros(n)

c = cov_loc(n, h, kernel="exponential")

# START WITH DETERMINISTIC TIKHONOV, ALSO TO DETERMINE ALPHA(DELTA)
options = {"parallel": use_ray, "alpha": alpha0, "delta": delta, "tau": tau}
x_tik, alpha_delta = solve("iterative_tikhonov", "deterministic", fwd, y, mean=prior_mean, options=options)

im_tik = np.reshape(x_tik, (n1, n2))
options["alpha"] = alpha_delta

# NEXT, COMPUTE NYS-EKI ESTIMATES FOR DIFFERENT VALUES OF J

options["sampling"] = "nystroem"
j_values = [100, 500, 1000, 2000, 3000, 5000, 8000]
nys_images = []
for j in j_values:
    options["j"] = j
    x_nys = solve("tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)
    im_nys = np.reshape(x_nys, (n1, n2))
    nys_images.append(im_nys)


plt.imshow(im_tik, cmap="gray")
plt.savefig("out/tik.png", bbox_inches='tight')

k=0
for j in j_values:
    plt.imshow(nys_images[k], cmap="gray")
    plt.savefig(f"out/nys_{j}.png", bbox_inches='tight')
    k += 1