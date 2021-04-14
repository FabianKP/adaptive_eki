
from math import sqrt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, downscale_local_mean

from inversion import *


# YOU CAN ADAPT THESE PARAMETERS
use_ray = True
n_alpha = 30
j = 1500    # 2000
snr = 10.  # desired signal-to-noise ratio
alpha_0 = 1.   # minimal alpha
c = 0.8 # sampling error is computed for alpha, alpha/c, alpha/c^2, ...
h = 1e-3

scaling_factor = 0.1 # determines the dimension of the problem

# GENERATE DATA
true_image = shepp_logan_phantom()
true_image = rescale(true_image, scale=scaling_factor, mode='reflect', multichannel=False)
n1, n2 = true_image.shape
theta = np.linspace(0., 180., max(n1, n2), endpoint=False)
y_true = radon(true_image, theta=theta, circle=False).flatten()

# GENERATE NOISY MEASUREMENT
m = y_true.size
sigma = np.linalg.norm(y_true) / (snr * sqrt(m))
noise = sigma * np.random.randn(*y_true.shape)
y = y_true + noise

# DISPLAY SOME INFOS
print("Actual signal-to-noise ratio: ", np.linalg.norm(y_true) / np.linalg.norm(noise))
n = n1 * n2
print(f"Parameter dimension n: {n}")
print(f"Measurement dimension m: {m}")

# NORMALIZE MEASUREMENT AND OPERATOR
scale = sigma * sqrt(m)
y = y / scale
def fwd(u):
    u_img = np.reshape(u, (n1, n2))
    y_img = radon(u_img, theta=theta, circle=False)
    return y_img.flatten() / scale

# create list of alphas
alphas = [alpha_0]
alpha = alpha_0
for i in range(n_alpha-1):
    alpha = alpha * c
    alphas.append(alpha)

prior_mean = np.zeros(n)
cov = cov_loc(n, h, kernel="exponential")

# COMPUTE TIKHONOV ESTIMATES
options = {"parallel": use_ray, "alpha_list": alphas, "return_list": True, "j": j}
traj_tik = solve("iterative_tikhonov", "deterministic", fwd, y, mean=prior_mean, cov=cov, options=options)

# COMPUTE EKI with 3 types of covariance approximation
options["sampling"] = "ensemble"
traj_eki  = solve("iterative_tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=cov, options=options)
options["sampling"] = "nystroem"
traj_nys = solve("iterative_tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=cov, options=options)
options["sampling"] = "svd"
traj_svd = solve("iterative_tikhonov", "ensemble", fwd, y, mean=prior_mean, cov=cov, options=options)

# COMPUTE APPROXIMATION ERRORS
traj_eki = np.array(traj_eki)
traj_nys = np.array(traj_nys)
traj_svd = np.array(traj_svd)

x_true = true_image.flatten()
true_norm = np.linalg.norm(x_true)
e_eki = np.linalg.norm(traj_eki - traj_tik, axis=1) / true_norm
e_nys = np.linalg.norm(traj_nys - traj_tik, axis=1) / true_norm
e_svd = np.linalg.norm(traj_svd - traj_tik, axis=1) / true_norm

# PLOTTING
plt.plot(alphas, e_eki, 'ro--', label="EKI")
plt.plot(alphas, e_nys, 'bx--', label="Nyström-EKI")
plt.plot(alphas, e_svd, 'gv--', label="SVD-EKI")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$e_\mathrm{app}$")
plt.legend(loc="upper right")
plt.savefig("out/divergence_wrt_alpha.png", bbox_inches='tight')
