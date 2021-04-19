"""
Contains the function 'ornstein_uhlenbeck' and auxiliary functions.
"""

import math
import numpy as np


def ornstein_uhlenbeck(n1, n2, h=0.1):
    """
    Computes the Ornstein-Uhlenbeck covariance matrix with given size and correlation length.
    The entries of the matrix are given by
    cov[i,j] = exp(-||pos(i) - pos(j)||/h),
    where pos(i) is the normalized position of the i-th pixel.
    :param n1: An integer. The image is assumed to have shape (n1, n2).
    :param n2: An integer. The image is assumed to have shape (n1, n2).
    :param h: The correlation length. A float. Small h corresponds to the assumption that distant pixels are
              uncorrelated.
    :return: The Ornstein-Uhlenbeck covariance matrix, a numpy array of shape (n1*n2, n1*n2).
    """
    # This function is simply a vectorized implementation of the above index-wise formula.
    # First, we compute the vector of normalized positions for all n*n-1 pixels.
    p = np.zeros((n1*n2, 2))
    for i in range(n1*n2):
        p[i, :] = _pos(i, n1, n2)
    pdiff0 = np.subtract.outer(p[:,0], p[:,0])
    pdiff1 = np.subtract.outer(p[:,1], p[:,1])
    pdiff = np.dstack((pdiff0, pdiff1))
    diffnorm = np.linalg.norm(pdiff, axis=2)
    cov = np.exp(-diffnorm / h)
    return cov

def _pos(i, n1, n2):
    """
    Given a picture of size (n,n), assuming that the pixels are in lexicographic order, and that the image is square.
    Returns an approximate position for each pixel, scaling the image to [0,1]^2.
    That is, the i-th pixel has the position [(i % n2) / (n2-1), (i // n2) / (n1-1)] in the domain [0,1]x[0,1].
    :param i: The index of the pixel, in lexicographic order. An integer between 0 and n1*n2-1.
    For example, i=n2-1 corresponds to the pixel at the upper right corner of the image.
    :return: The normalized position as a numpy vector of size 2.
    """
    x_position = (i % n2) / (n2-1)
    y_position = (i // n2) / (n1-1)
    return np.array([x_position, y_position])