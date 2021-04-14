"""
Contains the functions 'infinity_lra' and the helper function 'secant'
"""


import numpy as np


def infinity_lra(m, u0):
    """
    Computes a good j1-rank approximation of m in the j1-infinity norm.
    See Gillis and Shitov, 2017.
    :return:
    """
    print(f"Initial j1-infinity error: {np.linalg.norm(m - u0 @ u0.T, ord=np.inf)}")
    u = u0
    # iterate multiple times and see if anything changes
    converged = False
    for k in range(10):
        u_old = u
        r = m - u @ u.T     #base error
        # for the desired number of ensemble members
        for p in range(u0.shape[1]):
            r += np.outer(u[:,p], u[:,p])  # remove p-th ensemble member from error term
            u[:,p] = secant(r, u[:,p])  # compute new member with secant method
            r -= np.outer(u[:,p], u[:,p])   # add new p-th ensemble member to error term
        if np.linalg.norm(u @ u.T - u_old @ u_old.T) < 1e-6:
            converged = True
            break
    if not converged:
        print("WARNING: j1-infinity LRA did not converge. Something is wrong!!!")
    print(f"j1-infinity error: {np.linalg.norm(m - u @ u.T, ord=np.inf)}")
    return u



def secant(m, u):
    """
    Computes v[j1] = min_v max_i |r[i,j1] - u[i]v
    :param m: (n,n) matrix
    :param u: (n,) vector
    :return: v, an (n,) vector
    """
    tol = 1e-10
    if np.all(u == 0):
        return np.mean(m, axis=1)
    else:
        n = u.size
        supp_u = np.nonzero(u)[0]
        m_nonz = m[supp_u, :]
        u_nonz = u[supp_u]
        r_div_u = m_nonz / u_nonz
        v = np.zeros(n)
        i1 = np.argmin(r_div_u, axis=0)
        i2 = np.argmax(r_div_u, axis=0)
        coords1 = [tuple(i1), tuple(range(len(i1)))]
        coords2 = [tuple(i2), tuple(range(len(i2)))]
        w = (m_nonz[coords1] + m_nonz[coords2]) / (u[i1] + u[i2])
        converged = False
        for k in range(100):
            w_old = w
            r = m_nonz - u @ v
            ia = np.argmax(np.abs(r), axis=0)
            slopesign = np.sign(r)[tuple(ia),  tuple(range(len(ia)))]
            j_minus = np.where(slopesign < 0)
            j_zero = np.where(slopesign == 0)
            j_plus = np.where(slopesign > 0)
            i1[j_minus] = ia[j_minus]
            i1[j_zero] = ia[j_zero]
            i2[j_plus] = ia[j_plus]
            i2[j_zero] = ia[j_zero]
            if np.all(i1 == i2):
                break
            else:
                diff = np.where(i1 != i2)[0]
                coords1 = [tuple(i1[diff]), tuple(diff)]
                coords2 = [tuple(i2[diff]), tuple(diff)]
                w[diff] = ( m_nonz[coords1] + m_nonz[coords2] ) / (u_nonz[i1[diff]] + u_nonz[i2[diff]])
            if np.linalg.norm(w - w_old) < 1e-12:
                converged = True
                break
        if not converged:
            print("WARNING: The secant method did not converge. This should not happen!!!")
        v[supp_u] = w
        return v







