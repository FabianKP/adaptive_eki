
import numpy as np
import scipy.linalg as scilin
from time import time

from inversion.solvers import Solver


class Tikhonov(Solver):

    def solve(self):
        alpha = self.options.setdefault("alpha", 1.)
        a = self.mode.a()
        b = self._b(a)
        btb = b.T @ b
        rhs = b.T @ (self.y - self.fwd(self.mean))
        w = scilin.solve(btb + alpha * np.identity(btb.shape[0]), rhs, assume_a='pos')
        u = self.mean + a @ w
        return u