# RESULT: NYSTROEM WITH SVD IS BETTER

import numpy as np
import scipy.linalg as scilin
from matplotlib import pyplot as plt
from time import time

from inversion import ornstein_uhlenbeck

d = 80
h = 0.01

def nystroem_cholesky(a, j):
    x = np.random.randn(a.shape[0], j)
    y = a @ x
    q, r = np.linalg.qr(y)
    b1 = a @ q
    b2 = q.T @ b1
    c = np.linalg.cholesky(b2)
    ft = scilin.solve_triangular(c.T, b1.T)
    return ft.T

def nystroem_svd(a, j):
    """
    Computes a factorized low-rank approximation using the Nyström method, such that
    a_nys a_nys.T =  _c0 * q * (q.T * _c0 * q)^(-1) * q.T * _c0,
    where q is a suitable sketching matrix.
    """
    eps = 0.0
    x = np.random.randn(a.shape[0], j)
    y = a @ x
    q, r = np.linalg.qr(y)
    b1 = a @ q
    b2 = q.T @ b1
    d, u = scilin.eigh(b2)
    dclip = d.clip(min=eps)
    sqrtinv_qtcq = (u * np.divide(1, np.sqrt(dclip), out=np.zeros_like(dclip), where=dclip != 0))
    f = b1 @ sqrtinv_qtcq
    return f

def t_svd(s, u, j):
    st = u[:, -j:] * np.sqrt(s[-j:])
    return st

c0 = ornstein_uhlenbeck(d, d, h)
# compute svd of c0
print("Computing SVD")
s, u =  scilin.eigh(c0)
print("done")
# set sample sizes
j_list = [100, 200, 500, 1000, 2000, 3000, 4000, 5000]
# initialize error list
e_nyschol = []
e_nyssvd = []
e_svd = []
# define error function
def error(a):
    return np.linalg.norm(a @ a.T - c0)
for j in j_list:
    print(f"J={j}")
    t0 = time()
    a_nyschol = nystroem_cholesky(c0, j)
    t1 = time()
    print(f"Time for nyschol: {t1-t0}")
    a_nyssvd = nystroem_svd(c0, j)
    t2 = time()
    print(f"Time for nyssvd: {t2-t1}")
    a_svd = t_svd(s, u, j)
    e_nyschol.append(error(a_nyschol))
    e_nyssvd.append(error(a_nyssvd))
    e_svd.append(error(a_svd))
print("Done.")
# plot the results
plt.plot(j_list, e_nyschol, 'ro--', label="Nyström+Cholesky")
plt.plot(j_list, e_nyssvd, 'bx--', label="Nyström+SVD")
plt.plot(j_list, e_svd, 'gv--', label="Truncated SVD")
plt.xlabel("J")
plt.ylabel("error")
plt.legend(loc="upper right")
plt.show()

