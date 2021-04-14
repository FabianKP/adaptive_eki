
from math import log
import numpy as np


def fgd(c, u0, lam, eta, tau, maxiter):
    """
    Implements the factored gradient descent iteration described in Bhojanapalli (2016).
    :param x:
    :return:
    """
    # initialize
    u = u0
    r = c - u @ u.T
    error = np.linalg.norm(r, ord=1)
    print(f"l1 error: {error}")
    print(f"l2 error: {np.linalg.norm(r)}")
    # iterate until convergence
    for k in range(maxiter):
        print("Iteration ", k+1)
        u = u - eta * objective_grad(u @ u.T, c, u, tau, lam)
        print(f"Objective: {objective(u @ u.T, c, tau, lam)}")
        r = c - u @ u.T
        error = np.linalg.norm(r, ord=1)
        print(f"l1 error: {error}")
        print(f"l2 error: {np.linalg.norm(r)}")
        if error < 0.1:
            break
    return u


def objective(x, c, tau, lam):
    """
    Evaluates the objective function
    f(x) = sigma(c0 - x, tau) + lam*np.norm(x)**2
    :param x: a symmetric positive (n,n) array
    :param tau: a positive scalar
    :param lam: the regularization parameter, a positive scalar
    :return: the value of f at x, given the parameters tau and lam
    """
    return logsumexp(x-c, tau) + lam * np.linalg.norm(x)**2


def objective_grad(x, c, u, tau, lam):
    """
    Evaluates the gradient of the objective function f
    :param x:
    :param c:
    :param tau:
    :param lam:
    :return:
    """
    grad_first = u * logsumexpgrad(x-c, tau)[:, np.newaxis]
    grad_second = lam * x @ u
    return grad_first + grad_second


def logsumexp(x, tau):
    """
    Implements the logsumexp function from Kyrillidis (2018).
    :param x: an (n,n) matrix
    :param tau: a small positive scalar
    :return: sigma(x, tau), a scalar
    """
    y = x / tau
    p = np.exp(np.sum(y, axis=0)) + np.exp(-np.sum(y, axis=0))
    sigma = tau * log (np.sum(p) / (2*x.shape[0]**2))
    return sigma


def logsumexpgrad(x, tau):
    """
    Returns the gradient of the logsumexp function.
    :param x: an (n,n) matrix
    :param tau: tau: a small positive
    :return: grad_sigma, an (n,n) matrix
    """
    y = x / tau
    p = np.exp(np.sum(y, axis=0)) + np.exp(-np.sum(y, axis=0))
    grad_sigma = p / log(np.sum(p))
    return grad_sigma

