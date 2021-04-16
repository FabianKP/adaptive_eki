"""
Contains the functions 'tikhonov_list' and 'eki_list' with corresponding classes.
"""

import numpy as np
import scipy.linalg as scilin

from inversion.solver import ClassicSolver, EnsembleSolver


def tikhonov_list(fwd, y, x0, c0, alphas, options):
    """
    Computes Tikhonov-regularized solutions for a list of regularization parameters alpha
    :param fwd: The forward operator
    :param y: The measurement.
    :param x0: Initial guess.
    :param c0: Prior covariance.
    :param alphas: List of regularization parameters.
    :param options:
    :return: List of Tikhonov-regularized solutions corresponding to alphas.
    """
    tiklist = TikhonovList(fwd, y, x0, c0, alphas, options)
    return tiklist.solutions()


def eki_list(fwd, y, x0, c0, alphas, options):
    """
    Computes EKI estimates for a list of regularization parameters alpha
    :param fwd: The forward operator
    :param y: The measurement.
    :param x0: Initial guess.
    :param c0: Prior covariance.
    :param alphas: List of regularization parameters.
    :param options:
    :return: List of EKI estimates corresponding to alphas.
    """
    ekilist = EKIList(fwd, y, x0, c0, alphas, options)
    return ekilist.solutions()


class TikhonovList(ClassicSolver):

    def __init__(self, fwd, y, x0, c0, alphas, options):
        ClassicSolver.__init__(fwd, y, x0, c0, options)
        self._alphas = alphas

    def solutions(self):
        b = self._b(self._s)
        return _compute_solutions(a=self._s, b=b, y=self._y, fwd=self._fwd, x0=self._x0, alphas=self._alphas)


class EKIList(EnsembleSolver):

    def __init__(self, fwd, y, x0, c0, alphas, options):
        EnsembleSolver.__init__(fwd, y, x0, c0, options)
        self._alphas = alphas

    def solutions(self):
        b = self._b(self._a)
        return _compute_solutions(a=self._s, b=b, y=self._y, fwd=self._fwd, x0=self._x0, alphas=self._alphas)

def _compute_solutions(a, b, y, fwd, x0, alphas):
    btb = b.T @ b
    s, u = scilin.eigh(btb)
    au = a @ u
    utbt = u @ b.T
    rhs = utbt @ (y - fwd(x0))
    trajectory = []
    for alpha in alphas:
        print("alpha ", alpha)
        x_alpha = x0 + (au * np.divide(1, s + alpha)) @ rhs
        trajectory.append(x_alpha)
    return trajectory





