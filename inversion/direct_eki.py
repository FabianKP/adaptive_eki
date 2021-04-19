"""
Contains the function "direct_eki", an interface to the class "DirectEKI".
"""

from inversion.solver import EnsembleSolver


def direct_eki(fwd, y, x0, c0, alpha, options):
    """
    Computes the direct-EKI estimate
    x_alpha = x0 + a * (b.T @ b + alpha*identity)^(-1) * b.T * (y - fwd(x0)).
    :param fwd: The forward operator. Should map vectors of size n to vectors of size m.
    :param y: The measurement. A numpy vector of size m.
    :param x0: The initial guess. A numpy vector of size n.
    :param c0: The prior covariance. A numpy array of shape (n,n).
    :param alpha: The regularization parameter. A positive float.
    :param options: Possible options are:
        j - The sample size. An integer.
        parallel - A Boolean variable. If True, the Ray framework is used to parallelize forward operator evaluations.
    :return: x_alpha as given above. A numpy vector of size n.
    """
    deki = DirectEKI(fwd, y, x0, c0, alpha, options)
    x_hat = deki.solve()
    return x_hat


class DirectEKI(EnsembleSolver):
    """
    This class implements the direct EKI method.
    """
    def __init__(self, fwd, y, x0, c0, alpha, options):
        EnsembleSolver.__init__(self, fwd, y, x0, c0, options)
        self._alpha = alpha

    def solve(self):
        """
        Computes the direct-EKI estimate. Simply calls the _regularized_solution-function of the superclass 'Solver'
        with suitable input.
        :return: The solution, a numpy vector of size n.
        """
        return self._regularized_solution(self._a(), self._alpha)