
import numpy as np
from scipy.linalg import eigh, eig_banded


class Solver:
    def __init__(self, mode, fwd_operator, y, mean, cov, options=None):
        cov, options = self._handle_input(y, mean, cov, options)
        self.fwd = fwd_operator
        self.y = y
        self.mean = mean
        self.n = len(mean)
        self.m = len(y)
        self.c = cov
        self.options = options
        parallel = options.setdefault("parallel", False)
        if parallel:
            print("Using parallelization with Ray")
            from inversion.parallel_evaluations import _b_parallel
            def _b(a):
                return _b_parallel(self.fwd, a)
            self._b = _b
        if mode == "deterministic":
            self.mode = DeterministicMode(cov, options)
        else:
            self.mode = EnsembleMode(mean, cov, options)

    def _handle_input(self, y, mean, cov, options):
        # check that everything is what it should be
        if cov is None:
            cov = np.identity(mean.size)
        if options is None: options = {}
        else:
            assert isinstance(options, dict)
        return cov, options

    def _b(self, a):
        # preallocate
        b = np.zeros((self.m, a.shape[1]))
        # fill column-wise
        for i in range(1, a.shape[1]):
            b[:, i] = self.fwd(a[:, i])
        return b

    def _r(self, a, alpha=1., return_svd=False):
        """
        computes the transformation matrices sqrt_r = (a + alpha*I)^{-1/2} and r = (a + I)^(-1)
        using eigen decomposition: if a= u s u.T, then sqrt_r = u (s+alpha)^(-1/2) u.T
        and r = u (s+alpha)^(-1) u.T
        :param a: symmetric positive semidefinite matrix
        :param alpha: can also be a numpy vector, corresponding to the diagonal entries of the regularizer
        :param return_svd: if True, the method doesn't return T_root and T, but instead returns the SVD s, U,
        so that it can be reused for different values of alpha. In this case, the parameter alpha is ignored.
        :return: sqrt_r, r
        """
        s, u = eigh(a)
        if return_svd:
            return s, u
        else:
            # we now have B.T @ B = U diagflat(s) V
            sqrt_r = (u * np.divide(1., np.sqrt(s + alpha))) @ u.T
            r = (u * np.divide(1., (s + alpha))) @ u.T
            return sqrt_r, r

    def solve(self):
        raise NotImplementedError


class DeterministicMode:

    def __init__(self, cov, options=None):
        d, u = eigh(cov)
        dpos = d.clip(min=0.)
        self.s = u * np.sqrt(dpos)

    def a(self):
        return self.s


class EnsembleMode:

    def __init__(self, mean, cov, options=None):
        self.mean = mean
        self.n = cov.shape[1]
        self.cov = cov
        self.epsilon = "auto"
        self.j = options.setdefault("j1", 100)
        self.sampling = options.setdefault("sampling", "ensemble")
        if self.sampling == "ensemble":
            self._init_ensemble(cov)
        elif self.sampling == "svd":
            self._init_svd(cov)

    def a(self):
        if self.sampling == "ensemble":
            ensemble = self._generate_ensemble()
            a = self._anomaly(ensemble)
        elif self.sampling == "nystroem":
            a = self._nystroem()
        elif self.sampling == "svd":
            a = self._truncated_svd()
        else:
            raise NotImplementedError
        return a

    def _init_ensemble(self, cov):
        d, u = eigh(cov)
        dclip = d.clip(min=0.0)
        self.s = u * np.sqrt(dclip)

    def _init_svd(self, cov):
        d, u = eigh(cov)
        self.eigvals = d.clip(min=0.0)
        self.eigvecs = u

    def _generate_ensemble(self):
        """
        generate ensemble of size 'n_ensemble' with distribution normal(0,S @ S.T)
        :return: ensemble in a numpy matrix
        """
        # obtain state dimension
        return self.s @ np.random.randn(self.n, self.j)

    def _anomaly(self, ensemble):
        # computes anomaly of given ensemble X
        mean = np.mean(ensemble, axis=1)
        anomaly = (ensemble - mean[:,np.newaxis])/np.sqrt(self.j)
        return anomaly

    def _nystroem(self):
        eps = 0.0001
        x = np.random.randn(self.n, self.j)
        y = self.cov @ x
        q, r = np.linalg.qr(y)
        d, u = eigh(q.T @ self.cov @ q)
        dclip = d.clip(min=eps)
        sqrtinv_qtcq = (u * np.divide(1, np.sqrt(dclip), out=np.zeros_like(dclip), where=dclip != 0))
        a_nys = self.cov @ q @ sqrtinv_qtcq
        return a_nys

    def _truncated_svd(self):
        s = self.eigvecs[:, -self.j:] * np.sqrt(self.eigvals[-self.j:])
        return s


    # DEPRECATED

    def _init_tridiagonal(self):
        if self.epsilon == "auto":
            if self.j < self.n:
                epsilon = np.sqrt((self.n-self.j)/(self.n))
            else:
                epsilon = 0
        else:
            epsilon = self.epsilon
        nu = epsilon / 10
        b1 = np.ones(self.n)
        b2 = np.random.normal(loc=epsilon, scale=nu, size=self.n)  # good values are loc=0.2, scale=0.05#
        band = np.vstack((b1, b2))
        eigvals, eigvecs = eig_banded(band, lower=True)
        self.eigvals = eigvals.clip(min=0.)
        self.eigvecs = eigvecs

    def _tridiagonal_sampling(self):
        s = self.eigvecs[:, -self.j:] * np.sqrt(self.eigvals[-self.j:])
        return s

