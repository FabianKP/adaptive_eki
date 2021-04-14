

import numpy as np
import scipy.linalg as scilin

from inversion.solvers import Solver


class AdaptiveEKI(Solver):

    def solve(self):
        maxiter = self.options.setdefault("maxiter", 100)
        alpha = self.options.setdefault("alpha", 1.)
        delta = self.options["delta"]
        sampling = self.options["sampling"]
        c = self.options.setdefault("c", 0.8)
        tau = self.options.setdefault("tau", 1.)
        j0 = self.mode.j
        x = self.mean
        x_list = []
        # the actual computation starts
        for k in range(maxiter):
            print("Iteration ", k + 1)
            print("Ensemblesize: ", self.mode.j)
            a = self.mode.a()
            b = self._b(a)
            btb = b.T @ b
            rhs = b.T @ (self.y - self.fwd(self.mean))
            w = scilin.solve(btb + alpha*np.identity(btb.shape[0]), rhs, assume_a='pos')
            x = self.mean + a @ w
            x_list.append(x)
            # check discrepancy
            discrepancy = np.linalg.norm(self.y - self.fwd(x))
            print("Alpha: ", alpha)
            print("Discrepancy: ", discrepancy)
            if discrepancy < tau * delta:
                break
            else:
                alpha *= c
                if sampling == "ensemble":
                    self.mode.j = np.ceil(j0 / (alpha**2)).astype(int)
                else:
                    self.mode.j = np.ceil(j0 / alpha).astype(int)
                # if the sample size is larger than the state dimension, it makes no sense.
                if self.mode.j >= self.mean.size:
                    break
        return x, x_list