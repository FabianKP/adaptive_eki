"""
Contains the function "tikhonov", an interface to the class "Tikhonov".
"""

from inversion.solver import ClassicSolver


def tikhonov(fwd, y, x0, c0_root, alpha, options):
    tik = Tikhonov(fwd, y, x0, c0_root, alpha, options)
    x_alpha = tik.solve()
    return x_alpha


class Tikhonov(ClassicSolver):

    def __init__(self, fwd, y, x0, c0_root, alpha, options):
        ClassicSolver.__init__(self, fwd, y, x0, c0_root, options)
        self._alpha = alpha

    def solve(self):
        return self._regularized_solution(self._s, self._alpha)