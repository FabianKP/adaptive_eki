import numpy as np
import ray
ray.shutdown()
ray.init()

# from utils import isMatrix

def _b_parallel(fwd, a):
    @ray.remote
    def l(x):
        y = fwd(x)
        return y
    b_ray = [l.remote(a) for a in a.T]
    b = np.asarray(ray.get(b_ray)).T
    return b

