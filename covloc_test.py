# Some tests and visualizations of the function 'cov_loc'

import matplotlib.pyplot as plt
import scipy.linalg as scilin

from inversion.ornstein_uhlenbeck import ornstein_uhlenbeck


n = 1000
sigma = 1e-6

p = ornstein_uhlenbeck(n, sigma)

d, u = scilin.eigh(p)


plt.imshow(p)
plt.show()

