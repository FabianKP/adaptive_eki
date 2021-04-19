"""
Contains function 'matrix_sqrt'
"""
import numpy as np
import scipy.linalg as scilin

def matrix_sqrt(a):
    """
    Computes a matrix square-root.
    :param a: A positive Hermitian matrix.
    :return: s, evals, evecs. Here, s is a matrix such that s * s.T = a. The vector evals contains the eigenvectors of a,
    and the matrix evecs the corresponding eigenvectors.
    """
    # compute svd
    evals, evecs = scilin.eigh(a)
    evals = evals.clip(min=0.0)
    s = evecs * np.sqrt(evals)
    return s, evals, evecs
