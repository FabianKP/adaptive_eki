"""
Contains the class 'Solver'
"""


import numpy as np
import scipy.linalg as scilin


class Solver:
    def __init__(self, fwd, y, x0, c0, options=None):
        c0, options = self._handle_input(x0, c0, options)
        self._fwd = fwd
        self._y = y
        self._x0 = x0
        self._n = len(x0)
        self._m = len(y)
        self._c0 = c0
        self._options = options
        parallel = options.setdefault("parallel", False)
        if parallel:
            print("Using parallelization with Ray")
            from inversion.parallel_evaluations import _b_parallel
            def _b(a):
                return _b_parallel(self._fwd, a)
            self._b = _b

    def _handle_input(self, x0, cov, options):
        """
        If cov is None, it is replaced by the default np.identity(x0.size).
        Furthermore, if options is None, it is set to the empty dictionary.
        """
        if cov is None:
            cov = np.identity(x0.size)
        if options is None:
            options = {}
        else:
            assert isinstance(options, dict)
        return cov, options

    def _b(self, a):
        """
        Computes the matrix b = fwd(a), where fwd(a) is evaluated column-wise.
        :param a: A matrix of shape (n,j).
        :return: A matrix of shape (n,j).
        """
        b = np.zeros((self._m, a.shape[1]))
        # fill column-wise
        for i in range(1, a.shape[1]):
            b[:, i] = self._fwd(a[:, i])
        return b

    def _regularized_solution(self, a, alpha):
        """
        Computes the regularized solution
        x_alpha = x0 + s (b.T * b + alpha*identity)^(-1) * b.T * (y - fwd(x0))
        :param alpha: The regularization parameter, should be a strictly positive float.
        :return: The regularized solution
        """
        b = self._b(a)
        btb = b.T @ b
        rhs = b.T @ (self._y - self._fwd(self._x0))
        w = scilin.solve(btb + alpha * np.identity(btb.shape[0]), rhs, assume_a='pos')
        x_alpha = self._x0 + a @ w
        return x_alpha


class ClassicSolver(Solver):

    def __init__(self, fwd, y, x0, c0, options=None):
        Solver.__init__(self, fwd, y, x0, c0)
        # compute square-root of c0
        print("Initializing classic solver")
        d, u = scilin.eigh(c0)
        dpos = d.clip(min=0.)
        self._s = u * np.sqrt(dpos)
        print("Done.")


class EnsembleSolver(Solver):

    def __init__(self, fwd, y, x0, c0, options=None):
        Solver.__init__(self, fwd, y, x0, c0)
        self._j = options.setdefault("j", 100)
        self._sampling = options.setdefault("sampling", "standard")
        if self._sampling == "standard":
            self._init_ensemble()
        elif self._sampling == "nystroem":
            self._init_nystroem(c0)
        elif self._sampling == "svd":
            self._init_svd(c0)
        else:
            raise NotImplementedError

    def _a(self):
        """
        Computes the factor a of the low-rank approximation a * a.T of self._c0.
        :return: The matrix a with shape (self._c0.shape[0], self._j).
        """
        if self._sampling == "standard":
            ensemble = self._generate_ensemble()
            a = self._anomaly(ensemble)
        elif self._sampling == "nystroem":
            a = self._nystroem()
        elif self._sampling == "svd":
            a = self._tsvd()
        else:
            raise NotImplementedError
        return a

    def _init_ensemble(self):
        """
        Initializes the ensemble-based sampling, by computing the symmetric square-root _s of _c0.
        That is, this method sets the attribute _s such that _s * _s.T = _c0.
        """
        d, u = scilin.eigh(self._c0)
        dclip = d.clip(min=0.0)
        self._s = u * np.sqrt(dclip)

    def _init_nystroem(self, cov):
        """
        Computes a factorized low-rank approximation using the Nyström method, such that
        a_nys a_nys.T =  _c0 * q * (q.T * _c0 * q)^(-1) * q.T * _c0,
        where q is a suitable sketching matrix.
        Sets the attribute a_nys, which is a (n,n)-matrix.
        """
        x = np.random.randn(self._n, self._j)
        y = self._c0 @ x
        q, r = np.linalg.qr(y)
        b1 = self._c0 @ q
        b2 = q.T @ b1
        c = np.linalg.cholesky(b2)
        ft = scilin.solve_triangular(c.T, b1.T)
        self._a_nys = ft.T

    def _init_svd(self, cov):
        """
        Initializes the SVD-based sampling, by computing the singular value decomposition of _c0.
        Sets the attributes _eigvals and _eigvecs, where _eigvals is a vector that contains the eigenvalues of _c0 in
        ascending order, and _eigvecs is a matrix with the same shape as _c0, whose columns are given by the
        corresponding eigenvectors.
        """
        d, u = scilin.eigh(cov)
        self.eigvals = d.clip(min=0.0)
        self.eigvecs = u

    def _generate_ensemble(self):
        """
        Generates an ensemble of size 'j' with distribution normal(0,_s * _s.T)
        :return: A numpy matrix of size (_s.shape[0], _j).
        """
        return self._s @ np.random.randn(self._n, self._j)

    def _anomaly(self, ensemble):
        """
        Computes the anomaly of a given ensemble:
        anomaly = (ensemble - mean(ensemble)) / sqrt(j)
        :param ensemble: An (n,j)-matrix.
        :return: The ensemble anomaly, an (n,j)-matrix.
        """
        mean = np.mean(ensemble, axis=1)
        anomaly = (ensemble - mean[:,np.newaxis])/np.sqrt(self._j)
        return anomaly

    def _nystroem(self):
        """
        Returns a Nyström approximation of rank _j
        :return: A matrix with shape (_c0.shape[0], _j).
        """
        return self._a_nys[:,:self._j]

    def _tsvd(self):
        """
        Computes a factor st of the truncated singular value decomposition of _c0, i.e.
        st * st.T approximates _c0
        :return: A matrix of shape (_c0.shape[0], _j).
        """
        st = self.eigvecs[:, -self._j:] * np.sqrt(self.eigvals[-self._j:])
        return st
