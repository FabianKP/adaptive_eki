"""
Executing this file reproduces figures 1 and 3 of the paper.
The Radon transform of the Shepp-Logan phantom is computed, the measurement is contaminated with simulated noise,
and then the image is reconstructed from the noisy measurement using EKI.
"""

import numpy as np
import matplotlib.pyplot as plt

from inversion import *
from inversion.simulate_measurement import simulate_measurement

# YOU CAN ADAPT THESE PARAMETERS
use_ray = True          # use ray for parallelization to speed up the runtime
snr = 10.               # sets the signal-to-noise ratio for the simulated data
alpha0 = 1              # initial regularization parameter
j = 2000                # sample size
h = 1e-3                # correlation length for the prior covariance
scaling_factor = 0.15   # this factor determines the size of the used image and thereby the parameter and measurement dimension

# obtain data
y_hat_im, x_im, fwd, delta = simulate_measurement(snr, scaling_factor)
n_1, n_2 = x_im.shape
x = x_im.flatten()
y_hat = y_hat_im.flatten()
n = x.size
m = y_hat.size

# display basic info
print(f"Image size: {n_1}x{n_2}")
print(f"Parameter dimension: n={n}")
print(f"Measurement dimension: m={m}")
print(f"Initial ensemble size: j1={j}")

# create x0 and c0
x0 = np.zeros(n)
print("Computing covariance matrix")
c0 = ornstein_uhlenbeck(n, h)
print("Done.")

## RECONSTRUCT THE IMAGE FROM THE NOISY MEASUREMENT USING EKI:

# determine a good value for the regularization parameter alpha using the discrepancy principle
# and Tikhonov regularization
tau = 1.2
options = {"parallel": use_ray, "alpha": alpha0, "delta": delta, "tau": tau}
x_tik, alpha_delta = iterative_tikhonov(fwd=fwd, y=y_hat, x0=x0, c0=c0, delta=delta, options=options)

# next, compute EKI reconstructions first with Standard-EKI, then with Nyström-EKI
options["alpha"] = alpha_delta
options["j"] = j
options["sampling"] = "standard"
x_std = adaptive_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, delta=delta, options=options)
options["sampling"] = "nystroem"
x_nys = adaptive_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, delta=delta, options=options)

# plot the original image and the reconstructions
im_std = np.reshape(x_std, (n_1, n_2))
im_nys = np.reshape(x_nys, (n_1, n_2))
plt.imshow(x_im, cmap="gray")
plt.savefig("out/figure1.png", bbox_inches='tight')
plt.imshow(im_std, cmap="gray")
plt.savefig("out/figure3a.png", bbox_inches='tight')
plt.imshow(im_nys, cmap="gray")
plt.savefig("out/figure3b.png", bbox_inches='tight')
