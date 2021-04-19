"""
Executing this file reproduces figure 4 of the paper.
The Shepp-Logan phantom is reconstructed with Nyström-EKI from a noisy measurement with fixed regularization parameter
and varying sample sizes.
"""

import numpy as np
import matplotlib.pyplot as plt

from inversion import *

# YOU CAN ADAPT THESE PARAMETERS
use_ray = True
snr = 10.
alpha0 = 1.
tau = 1.2
h = 0.01
scaling_factor = 0.25
j_values = [100, 500, 1000, 2000, 3000, 5000, 8000]

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
x0 = np.zeros(n)
c0 = ornstein_uhlenbeck(n1, n2, h)
print("Computing SVD of c0...")
c0_root, evals, evecs = matrix_sqrt(c0)
print("...done.")

# determine a good value for the regularization parameter alpha using the discrepancy principle
# and Tikhonov regularization
options = {"parallel": use_ray, "alpha": alpha0, "delta": delta, "tau": tau}
x_tik, alpha = iterative_tikhonov(fwd=fwd, y=y_hat, x0=x0, c0_root=c0_root, delta=delta, options=options)
im_tik = np.reshape(x_tik[-1], (n1, n2))

# next, apply Nyström-EKI with different sample sizes
options["sampling"] = "nystroem"
nys_images = []
for j in j_values:
    options["j"] = j
    print(f"Sample size J={j}")
    x_nys = direct_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, alpha=alpha, options=options)
    im_nys = np.reshape(x_nys, (n1, n2))
    nys_images.append(im_nys)

# plot the results
plt.imshow(im_tik, cmap="gray")
plt.savefig("out/figure4_tik.png", bbox_inches='tight')
k=0
for j in j_values:
    plt.imshow(nys_images[k], cmap="gray")
    plt.savefig(f"out/figure4_nys_{j}.png", bbox_inches='tight')
    k += 1