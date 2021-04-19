"""
Contains the functions 'tikhonov_list' and 'eki_list' with corresponding classes.
"""

import numpy as np
import scipy.linalg as scilin

from inversion.solver import ClassicSolver, EnsembleSolver


def tikhonov_list(fwd, y, x0, c0_root, alphas, options):
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
    tiklist = TikhonovList(fwd, y, x0, c0_root, alphas, options)
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
    """
    Class underlying the function 'tikhonov_list'
    """
    def __init__(self, fwd, y, x0, c0_root, alphas, options):
        ClassicSolver.__init__(self, fwd, y, x0, c0_root, options)
        self._alphas = alphas

    def solutions(self):
        """
        This computes the Tikhonov-regularized solutions for the list of regularization parameters self._alphas.
        It simply calls the auxiliary "_compute_solutions" functions with a = self._s, where self._s is a matrix
        square root of the prior covariance matrix self._c0.
        :return: A list of numpy vectors.
        """
        b = self._b(self._s)
        return _compute_solutions(a=self._s, b=b, y=self._y, fwd=self._fwd, x0=self._x0, alphas=self._alphas)


class EKIList(EnsembleSolver):
    """
    Class underlying the function "eki_list".
    """
    def __init__(self, fwd, y, x0, c0, alphas, options):
        EnsembleSolver.__init__(self, fwd, y, x0, c0, options)
        self._alphas = alphas

    def solutions(self):
        """
        This computes the direct-EKI estimates for the list of regularization parameters self._alphas.
        It simply calls the auxiliary "_compute_solutions" functions with a = self._a().
        :return: A list of numpy vectors.
        """
        a = self._a()
        b = self._b(a)
        return _compute_solutions(a=a, b=b, y=self._y, fwd=self._fwd, x0=self._x0, alphas=self._alphas)


def _compute_solutions(a, b, y, fwd, x0, alphas):
    """
    An auxiliary function. Given the parameters, it computes a list of estimates
    solutions[i] = x0 + a * (b.T*b + alphas[i]*identity)^(-1) * b.T * (y - fwd(x0)),
    for alpha in alphas.
    :param a: A numpy array of shape (n,j).
    :param b: A numpy array of shape (m,j).
    :param y: A vector of size m.
    :param fwd: The forward operator. Should take vectors of size n as input.
    :param x0: Initial guess, a vector of size n.
    :param alphas: The list of regularization parameters.
    :return: Returns the list solutions, where solutions[i] is as above.
    """
    btb = b.T @ b
    s, u = scilin.eigh(btb)
    au = a @ u
    utbt = u @ b.T
    rhs = utbt @ (y - fwd(x0))
    solutions = []
    for alpha in alphas:
        print("alpha ", alpha)
        x_alpha = x0 + (au * np.divide(1, s + alpha)) @ rhs
        solutions.append(x_alpha)
    return solutions





