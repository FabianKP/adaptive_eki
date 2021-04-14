"""
Implements a localized covariance
"""

import math
import numpy as np


def cov_loc(n, sigma=0.1, kernel="exponential"):
    """
    Returns a localized covariance of shape (n,n)
    :param n: strictly positive integer
    :return:
    """
    cov = np.ones((n, n))
    if kernel == "tridiagonal":
        eps = np.sqrt(0.7)
        cov = np.diagflat(np.ones(n)) + eps*np.diagflat(np.ones(n-1),k=-1) + eps*np.diagflat(np.ones(n-1),k=1)
    else:
        for i in range(n):
            for j in range(n):
                cov[i,j] = correlation(pos(i, n), pos(j, n), sigma, kernel)
    return cov

def correlation(pos_1, pos_2, sigma, kernel):
    if kernel == "exponential":
        cor = math.exp(-np.linalg.norm(pos_1 - pos_2) / sigma)
    elif kernel == "gaussian":
        cor = math.exp(-np.linalg.norm(pos_1 - pos_2)**2 / sigma)
    else:
        raise NotImplementedError
    return cor

def pos(i, n):
    """
    Given a picture of size (n,n), assuming that the pixels are in lexicographic order, and that the image is square.
    Returns an approximate position for each pixel, scaling the image to [0,1]^2.
    :param i:
    :return:
    """
    x_position = (i % n) / (n-1)
    y_position = (i // n) / (n-1)
    return np.array([x_position, y_position])