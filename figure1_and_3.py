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
alpha0 = 1.              # initial regularization parameter
j = 2000               # sample size
h = 0.01               # correlation length for the prior covariance
scaling_factor = 0.25   # this factor determines the size of the used image and thereby the parameter and measurement dimension


def compute_figures1_and_3():
    """
    Performs the main computation.
    Creates csv-files 'out/figure[i].csv`, with i = 1, 3a, 3b, 3c, 3d.
    """
    y_im, y_hat_im, x_im, fwd, delta = simulate_measurement(snr, scaling_factor)
    n1, n2 = x_im.shape
    x = x_im.flatten()
    y_hat = y_hat_im.flatten()
    n = x.size
    m = y_hat.size

    # store the ground truth, the measurement, and the noisy data
    np.savetxt("out/figure1.csv", x_im, delimiter=",")
    np.savetxt("out/figure3a.csv", y_im, delimiter=",")
    np.savetxt("out/figure3b.csv", y_hat_im, delimiter=",")

    # display basic info
    print(f"Image size: {n1}x{n2}")
    print(f"Parameter dimension: n={n}")
    print(f"Measurement dimension: m={m}")
    print(f"Ensemble size: J={j}")

    # create x0 and c0
    x0 = np.zeros(n)
    c0 = ornstein_uhlenbeck(n1, n2, h)
    print("Computing SVD of c0...")
    c0_root, evals, evecs = matrix_sqrt(c0)
    print("...done.")

    ## RECONSTRUCT THE IMAGE FROM THE NOISY MEASUREMENT USING EKI:

    # determine a good value for the regularization parameter alpha using the discrepancy principle
    # and Tikhonov regularization
    tau = 1.2
    options = {"parallel": use_ray, "alpha": alpha0, "delta": delta, "tau": tau}
    x_tik, alpha_delta = iterative_tikhonov(fwd=fwd, y=y_hat, x0=x0, c0_root=c0_root, delta=delta, options=options)

    # next, compute EKI reconstructions first with Standard-EKI, then with Nystr??m-EKI
    options["j"] = j
    options["sampling"] = "standard"
    # standard sampling requires a matrix square root of c0
    options["c0_root"] = c0_root
    x_std = direct_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, alpha=alpha_delta, options=options)
    options["sampling"] = "nystroem"
    x_nys = direct_eki(fwd=fwd, y=y_hat, x0=x0, c0=c0, alpha=alpha_delta, options=options)

    # plot the original image and the reconstructions
    im_std = np.reshape(x_std, (n1, n2))
    im_nys = np.reshape(x_nys, (n1, n2))
    np.savetxt("out/figure3c.csv", im_std, delimiter=",")
    np.savetxt("out/figure3d.csv", im_nys,  delimiter=",")


def plot_figures1_and_3():
    """
    Creates plots from the corresponding csv_files and saves them under 'out/figure[i].png', with i = 1, 3a, 3b, 3c, 3d.
    """
    # We do not want axes for these images.
    plt.axis("off")
    for name in ["1", "3a", "3b", "3c", "3d"]:
        # Load array from .csv-files
        image = np.loadtxt(f"out/figure{name}.csv", delimiter=",")
        # Create and save plot
        plt.imshow(image, cmap="gray")
        plt.savefig(f"out/figure{name}.png", bbox_inches="tight", pad_inches=0)


compute_figures1_and_3()
plot_figures1_and_3()


