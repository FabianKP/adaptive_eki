"""
Contains the function 'solve'.
"""

from inversion.tikhonov import tikhonov
from inversion.direct_eki import direct_eki
from inversion.iterative_tikhonov import iterative_tikhonov
from inversion.adaptive_eki import adaptive_eki
from inversion.alpha_list import tikhonov_list, eki_list
from inversion.ornstein_uhlenbeck import ornstein_uhlenbeck
from inversion.simulate_measurement import simulate_measurement