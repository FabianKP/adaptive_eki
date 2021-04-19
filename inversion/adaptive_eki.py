"""
Contains the function 'adaptive_eki' and the accompanying class AdaptiveEKI.
"""

import numpy as np
import scipy.linalg as scilin

from inversion.solver import EnsembleSolver

def adaptive_eki(fwd, y, x0, c0, delta, options):
    """
    Interface to the AdaptiveEKI class.
    :param fwd: The forward operator.
    :param y: The measurement.
    :param x0: The initial guess.
    :param c0: The regularization matrix. Needs to have shape (n,n), where n is the size of x0.
    :param delta: The noise level.
    :param options:
        - maxiter: Maximum number of iterations
        - sampling: Type of sampling, see EnsembleSolver.
        - alpha1: The initial regularization parameter.
        - c: A constant that determines the sequence of regularization paramters. The regularization parameter
        alpha is updated by setting alpha = c*alpha.
        - tau: The 'fudge paramter' for the discrepancy principle. Should be larger than 1.
    :return trajectory, alpha: Returns the whole iteration as a list of numpy vectors. The last entry is the final estimate,
    which satisfies the discrepancy principle. Also returns the final regularization parameter alpha.
    """
    aeki = AdaptiveEKI(fwd, y, x0, c0, delta, options)
    trajectory, alpha = aeki.solve()
    return trajectory, alpha


class AdaptiveEKI(EnsembleSolver):
    """
    Implementation of the adaptive EKI method.
    """
    def __init__(self, fwd, y, x0, c0, delta, options):
        EnsembleSolver.__init__(self, fwd, y, x0, c0, options)
        self._delta = delta

    def solve(self):
        """
        Main routine. Computes the iterates of the adaptive EKI iteration and stops using the
        discrepancy principle.
        :return: The trajectory, a list of numpy vectors.
        """
        maxiter = self._options.setdefault("maxiter", 100)
        alpha = self._options.setdefault("alpha1", 1.)
        c = self._options.setdefault("c", 0.8)
        tau = self._options.setdefault("tau", 1.2)
        j1 = self._j
        trajectory = []
        # the actual computation starts
        for k in range(maxiter):
            print("Iteration ", k + 1)
            print("Sample size: ", self._j)
            # compute next step
            x_k = self._regularized_solution(self._a(), alpha)
            trajectory.append(x_k)
            # check discrepancy
            discrepancy = np.linalg.norm(self._y - self._fwd(x_k))
            print("alpha: ", alpha)
            print("Discrepancy: ", discrepancy)
            if discrepancy < tau * self._delta:
                break
            else:
                # decrease alpha and increase alpha accordingly
                alpha *= c
                if self._sampling == "standard":
                    self._j = np.ceil(j1 / (alpha ** 2)).astype(int)
                else:
                    # if SVD- or Nyström-based sampling is used, the sample size does not have to be increased as much
                    self._j = np.ceil(j1 / alpha).astype(int)
                # It makes no sense to continue the iteration when the sample size is larger than the
                # parameter dimension.
                if self._j >= self._x0.size:
                    break
        return trajectory, alpha