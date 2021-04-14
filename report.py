from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

from inversion import *

# YOU CAN ADAPT THESE PARAMETERS
use_ray = True
snr = 10.
alpha0 = 1
j0 = 20
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
n = n1 * n2

prior_mean = np.zeros(n)
c = cov_loc(n, h, kernel="exponential")

# ITERATIVE TIKHONOV REGULARIZATION
options = {"parallel": use_ray, "alpha": alpha0, "delta": delta, "tau": tau}
x_tik, alpha_delta = solve("iterative_tikhonov", "deterministic", fwd, y, mean=prior_mean, options=options)

# ADAPTIVE SVD-EKI
options["j"] = j0
options["sampling"] = "svd"
x_eki, list = solve("adaptive_eki", "ensemble", fwd, y, mean=prior_mean, cov=c, options=options)


# PLOTTING
im_tik = np.reshape(x_tik, (n1, n2))
im_eki = np.reshape(x_eki, (n1, n2))

plt.imshow(true_image, cmap="gray")
plt.savefig("out/report_shepp_logan.png", bbox_inches='tight')
plt.imshow(im_tik, cmap="gray")
plt.savefig("out/report_tikhonov.png", bbox_inches='tight')
plt.imshow(im_eki, cmap="gray")
plt.savefig("out/report_eki.png", bbox_inches='tight')