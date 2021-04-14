

import numpy as np

from inversion.solvers import Solver


class IterativeTikhonov(Solver):

    def solve(self):
        alpha_list = self.options.setdefault("alpha_list", None)
        maxiter = self.options.setdefault("maxiter", 100)
        if alpha_list is None:
            return_list = False
            alpha = self.options.setdefault("alpha", 1.)
            delta = self.options["delta"]
        else:
            return_list = True
            alpha = alpha_list[0]
            maxiter = len(alpha_list)
            delta = 0.
        c = self.options.setdefault("c", 0.8)
        tau = self.options.setdefault("tau", 1.5)
        # the actual computation starts
        a = self.mode.a()
        b = self._b(a)
        btb = b.T @ b
        s, u = self._r(btb, return_svd=True)
        utbt = u.T @ b.T
        au = a @ u
        rhs = utbt @ (self.y - self.fwd(self.mean))
        solutions = []
        for k in range(maxiter):
            print("Iteration ", k + 1)
            u = self.mean + (au * np.divide(1, s + alpha)) @ rhs
            solutions.append(u)
            # check discrepancy
            discrepancy = np.linalg.norm(self.y - self.fwd(u))
            print("Alpha: ", alpha)
            print("Discrepancy: ", discrepancy)
            if discrepancy < tau * delta:
                break
            if alpha_list is None:
                alpha *= c
            else:
                if k < maxiter-1: alpha = alpha_list[k+1]
        if return_list:
            return solutions
        else:
            return u, alpha