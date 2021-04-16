"""
Contains the function "direct_eki", an interface to the class "DirectEKI".
"""

from inversion.solver import EnsembleSolver


def direct_eki(fwd, y, x0, c0, alpha, options):
    deki = DirectEKI(fwd, y, x0, c0, alpha, options)
    x_hat = deki.solve()
    return x_hat


class DirectEKI(EnsembleSolver):

    def __init__(self, fwd, y, x0, c0, alpha, options):
        EnsembleSolver.__init__(self, fwd, y, x0, c0, options)
        self._alpha = alpha

    def solve(self):
        return self._regularized_solution(self._alpha)