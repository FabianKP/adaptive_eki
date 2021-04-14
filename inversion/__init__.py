"""
Contains the function 'solve'.
"""

from inversion.tikhonov import Tikhonov
from inversion.iterative_tikhonov import IterativeTikhonov
from inversion.adaptive_eki import AdaptiveEKI
from inversion.ornstein_uhlenbeck import ornstein_uhlenbeck
from inversion.simulate_measurement import simulate_measurement

def solve(iteration, mode, fwd_operator, y, mean, cov=None, options=None):
    if iteration == "tikhonov":
        solver = Tikhonov(mode, fwd_operator, y, mean, cov, options)
    elif iteration == "iterative_tikhonov":
        solver = IterativeTikhonov(mode, fwd_operator, y, mean, cov, options)
    elif iteration == "adaptive_eki":
        solver = AdaptiveEKI("ensemble", fwd_operator, y, mean, cov, options)
    else:
        raise NotImplementedError
    return solver.solve()