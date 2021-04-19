"""
Contains the function 'iterative_tikhonov' and the accompanying class.
"""

import numpy as np
import scipy.linalg as scilin

from inversion.solver import ClassicSolver


def iterative_tikhonov(fwd, y, x0, c0_root, delta, options):
    """
    Implements the iterative Tikhonov method, which Tikhonov solutions until the discrepancy principle is
    satisfied.
    :param fwd: The forward operator.
    :param y: The measurement.
    :param x0: The initial guess.
    :param c0_root: The square-root of the regularization matrix. Needs to have shape (n,n), where n is the size of x0.
    :param delta: The noise level.
    :param options:
        - maxiter: Maximum number of iterations
        - alpha1: The initial regularization parameter.
        - c: A constant that determines the sequence of regularization paramters. The regularization parameter
        alpha is updated by setting alpha = c*alpha.
        - tau: The 'fudge paramter' for the discrepancy principle. Should be larger than 1.
    :return trajectory: Returns the whole iteration as a list of numpy vectors. The last entry is the final estimate,
    which satisfies the discrepancy principle. Also returns the final regularization parameter alpha.
    """
    itik = IterativeTikhonov(fwd, y, x0, c0_root, delta, options)
    trajectory, alpha = itik.solve()
    return trajectory, alpha


class IterativeTikhonov(ClassicSolver):
    """
    Implementation of the iterative Tikhonov method.
    """
    def __init__(self, fwd, y, x0, c0_root, delta, options):
        ClassicSolver.__init__(self, fwd, y, x0, c0_root, options)
        self._delta = delta

    def solve(self):
        """
        Main routine of the IterativeTikhonov class.
        :return: The iterates of the iterative Tikhonov method as list of numpy vectors.
        """
        # load the tunable parameters from the options
        maxiter = self._options.setdefault("maxiter", 100)
        alpha1 = self._options.setdefault("alpha1", 1.)
        c = self._options.setdefault("c0", 0.8)
        tau = self._options.setdefault("tau", 1.5)
        # the actual computation starts
        alpha = alpha1
        b = self._b(self._s)
        btb = b.T @ b
        # In order to save computation time, we compute the singular value decomposition of b.T @ b once.
        # Then the Tikhonov regularized solution x = x0 + s * (b.T * b + alpha*identity)^(-1) * b.T * (y - fwd(x0))
        # can be computed very efficiently for different values of alpha.
        s, x_k = scilin.eigh(btb)
        utbt = x_k.T @ b.T
        su = self._s @ x_k
        rhs = utbt @ (self._y - self._fwd(self._x0))
        trajectory = []
        for k in range(maxiter):
            print("Iteration ", k + 1)
            # The next line is equivalent to
            # x_k = self._x0 + self._s @ np.inv(b.T @ b + alpha * identity(b.shape[1])) @ b.T @ (self._y - self._fwd(self._x0))
            # but uses the precomputed svd to be computationally more efficient.
            x_k = self._x0 + (su * np.divide(1, s + alpha)) @ rhs
            trajectory.append(x_k)
            # check whether the discrepancy principle is satisfied.
            discrepancy = np.linalg.norm(self._y - self._fwd(x_k))
            print("alpha: ", alpha)
            print("Discrepancy: ", discrepancy)
            if discrepancy < tau * self._delta:
                break
            else:
                # if the discrepancy principle is not satisfied, decrease alpha and repeat.
                alpha *= c
        return trajectory, alpha