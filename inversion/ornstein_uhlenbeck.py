"""
Contains the function 'ornstein_uhlenbeck' and auxiliary functions.
"""

import math
import numpy as np


def ornstein_uhlenbeck(n, h=0.1):
    """
    Computes the Ornstein-Uhlenbeck covariance matrix with size and correlation length
    :param n: the desired dimension.
    :param h: the correlation length; small h corresponds to the assumption that distant pixels are uncorrelated
    :return: cov - the Ornstein-Uhlenbeck covariance matrix, an ndarray of shape (n,n)
    """
    # compute all positions
    p = np.zeros((n, 2))
    for i in range(n):
        p[i, :] = _pos(i,n)

    pdiff0 = np.subtract.outer(p[:,0], p[:,0])
    pdiff1 = np.subtract.outer(p[:,1], p[:,1])
    pdiff = np.dstack((pdiff0, pdiff1))
    diffnorm = np.linalg.norm(pdiff, axis=2)
    cov = np.exp(-diffnorm / h)
    return cov

def _pos(i, n):
    """
    Given a picture of size (n,n), assuming that the pixels are in lexicographic order, and that the image is square.
    Returns an approximate position for each pixel, scaling the image to [0,1]^2.
    :param i: the index of the pixel in lexicographic order. For example, i=n-1 corresponds to the pixel at the
    upper right corner of the image
    :return: the normalized position as a numpy vector of shape (2,)
    """
    x_position = (i % n) / (n-1)
    y_position = (i // n) / (n-1)
    return np.array([x_position, y_position])