

import numpy as np
import scipy.linalg as scilin

from inversion.solver import ClassicSolver


def iterative_tikhonov(fwd, y, x0, c0, delta, options):
    """
    Implements the iterative Tikhonov method.
    :param fwd: The forward operator.
    :param y: The measurement.
    :param x0: The initial guess.
    :param c0: The regularization matrix. Needs to have shape (n,n), where n is the size of x0.
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
    itik = IterativeTikhonov(fwd, y, x0, c0, delta, options)
    trajectory, alpha = itik.solve()
    return trajectory, alpha


class IterativeTikhonov(ClassicSolver):

    def __init__(self, fwd, y, x0, c0, delta, options):
        ClassicSolver.__init__(self, fwd, y, x0, c0, options)
        self._delta = delta

    def solve(self):
        maxiter = self._options.setdefault("maxiter", 100)
        alpha1 = self._options.setdefault("alpha1", 1.)
        c = self._options.setdefault("c0", 0.8)
        tau = self._options.setdefault("tau", 1.5)
        # the actual computation starts
        alpha = alpha1
        b = self._b(self._s)
        btb = b.T @ b
        print("Start computing svd...")
        s, u = scilin.eigh(btb)
        print("done.")
        utbt = u.T @ b.T
        su = self._s @ u
        rhs = utbt @ (self._y - self._fwd(self._x0))
        trajectory = []
        for k in range(maxiter):
            print("Iteration ", k + 1)
            u = self._x0 + (su * np.divide(1, s + alpha)) @ rhs
            trajectory.append(u)
            # check discrepancy
            discrepancy = np.linalg.norm(self._y - self._fwd(u))
            print("alpha: ", alpha)
            print("Discrepancy: ", discrepancy)
            if discrepancy < tau * self._delta:
                break
            else:
                alpha *= c
        return trajectory, alpha