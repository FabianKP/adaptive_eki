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
    cov = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            cov[i,j] = math.exp(-np.linalg.norm(_pos(i, n) - _pos(j, n)) / h)
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