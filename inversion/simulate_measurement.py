"""
Contains the function simulate_measurement
"""

from math import sqrt
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

def simulate_measurement(snr, scaling_factor):
    """
    Generates the Radon transform of the Shepp-Logan phantom and adds simulated noise with given signal-to-noise ratio
    :param snr: the desired signal-to-noise ratio
    :param scaling_factor: a scaling factor that determines the size of the image, and thereby also the parameter
    and measurement dimension
    :return: y_hat_im, x_im, _fwd, delta
    where y_hat_im is the noisy measurement (in image format), x_im is the original image,
    _fwd is the flattened forward operator (i.e. the radon transform taking vectors as input and output)
    and delta is the noise level.
    """
    # load and rescale Shepp-Logan phantom
    x_im = rescale(shepp_logan_phantom(), scale=scaling_factor, mode='reflect', multichannel=False)
    n_1, n_2 = x_im.shape
    # compute the Radon transform of the Shepp-Logan phantom
    theta = np.linspace(0., 180., max(n_1, n_2), endpoint=False)
    y_im = radon(x_im, theta=theta, circle=False)
    m = y_im.size
    # create a noisy measurement with the given signal-to-noise ratio
    sigma = np.linalg.norm(y_im) / (snr * sqrt(m))
    standard_noise = np.random.randn(*y_im.shape)
    # rescale noise to ensure given snr
    noise = standard_noise * np.linalg.norm(y_im) / (snr * np.linalg.norm(standard_noise))
    y_hat_im = y_im + noise
    # rescale everything
    scale = sigma * sqrt(m)
    y_hat_im = y_hat_im / scale
    y_im = y_im / scale
    def fwd(u):
        u_img = np.reshape(u, (n_1, n_2))
        y_img = radon(u_img, theta=theta, circle=False)
        return y_img.flatten() / scale
    # compute delta
    delta = np.linalg.norm(y_hat_im - y_im)

    return y_im, y_hat_im, x_im, fwd, delta