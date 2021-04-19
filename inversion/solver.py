"""
Contains the class 'Solver'
"""


import numpy as np
import scipy.linalg as scilin


class Solver:
    def __init__(self, fwd, y, x0, options=None):
        options = self._handle_input(options)
        self._fwd = fwd
        self._y = y
        self._x0 = x0
        self._n = len(x0)
        self._m = len(y)
        self._options = options
        parallel = options.setdefault("parallel", False)
        if parallel:
            from inversion.parallel_evaluations import _b_parallel
            def _b(a):
                return _b_parallel(self._fwd, a)
            self._b = _b

    def _handle_input(self, options):
        """
        If cov is None, it is replaced by the default np.identity(x0.size).
        Furthermore, if options is None, it is set to the empty dictionary.
        """
        if options is None:
            options = {}
        else:
            assert isinstance(options, dict)
        return options

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

    def __init__(self, fwd, y, x0, c0_root, options=None):
        Solver.__init__(self, fwd, y, x0, options)
        # compute square-root of c0
        self._s = c0_root


class EnsembleSolver(Solver):

    def __init__(self, fwd, y, x0, c0, options=None):
        """
        :param fwd:
        :param y:
        :param x0:
        :param c0:
        :param options: A dict that provides additional solver options.
            -sampling: which type of sampling is used. Possible values are 'standard',
            'nystroem' and 'svd'. Default is 'nystroem'. If sampling is set to 'standard',
            the user has to provide the parameter c0_root.
            If sampling is set to 'svd', the user has to provide c0_eigvals and c0_eigvecs
        """
        Solver.__init__(self, fwd, y, x0, options)
        self._c0 = c0
        self._j = options.setdefault("j", 100)
        self._sampling = options.setdefault("sampling", "nystroem")
        if self._sampling == "standard":
            # for standard sampling, have to provide a square-root of c0
            self._c0_root = options["c0_root"]
        elif self._sampling == "nystroem":
            pass
        elif self._sampling == "svd":
            # for svd sampling, have to provide eigenvectors and eigenvals of c0
            self._c0_evals = options["c0_eigenvalues"]
            self._c0_evecs = options["c0_eigenvectors"]
        else:
            raise NotImplementedError

    def _a(self):
        """
        Computes the factor a of the low-rank approximation a * a.T of self._c0.
        :return: The matrix a with shape (self._c0.shape[0], self._j).
        """
        # the way in which a is generated depends on the chosen sampling scheme
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

    def _generate_ensemble(self):
        """
        Generates an ensemble of size 'j' with distribution normal(0,_s * _s.T)
        :return: A numpy matrix of size (_s.shape[0], _j).
        """
        return self._c0_root @ np.random.randn(self._n, self._j)

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
        Computes a factorized low-rank approximation using the Nyström method, such that
        a_nys a_nys.T =  _c0 * q * (q.T * _c0 * q)^(-1) * q.T * _c0,
        where q is a suitable sketching matrix.
        """
        x = np.random.randn(self._n, self._j)
        y = self._c0 @ x
        q, r = np.linalg.qr(y)
        d, u = scilin.eigh(q.T @ self._c0 @ q)
        dclip = d.clip(min=0.)  # removes negative eigenvalues that might have slipped in through numerical errors
        sqrtinv_qtcq = (u * np.divide(1, np.sqrt(dclip), out=np.zeros_like(dclip), where=dclip != 0))
        a_nys = self._c0 @ q @ sqrtinv_qtcq
        return a_nys

    def _tsvd(self):
        """
        Computes a factor st of the truncated singular value decomposition of _c0, i.e.
        st * st.T approximates _c0
        :return: A matrix of shape (_c0.shape[0], _j).
        """
        st = self._c0_evecs[:, -self._j:] * np.sqrt(self._c0_evals[-self._j:])
        return st
