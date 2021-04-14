# Try to find out in which way the new sampling scheme is better

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm, matrix_rank
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from scipy.linalg import eigh


from inversion.solvers import EnsembleMode



def setup_problem(scaling_factor, snr):
    use_ray = True  # don't use for actual experiments

    # GENERATE DATA
    true_image = shepp_logan_phantom()
    true_image = rescale(true_image, scale=scaling_factor, mode='reflect', multichannel=False)
    n1, n2 = true_image.shape
    theta = np.linspace(0., 180., max(n1, n2), endpoint=False)
    y_true = radon(true_image, theta=theta, circle=False).flatten()
    m = y_true.size
    sigma = np.linalg.norm(y_true) / (snr * sqrt(m))
    noise = sigma * np.random.randn(m)
    y = y_true + noise
    # rescale everything
    scale = sigma * sqrt(m)
    y = y / scale
    y_true = y_true / scale

    def fwd(u):
        u_img = np.reshape(u, (n1, n2))
        y_img = radon(u_img, theta=theta, circle=False)
        return y_img.flatten() / scale

    delta = np.linalg.norm(y - y_true)
    print(f"||y - y_true|| = {delta:6f}")

    n = n1 * n2

    x0 = np.zeros(n)
    s = np.identity(n)

    print(f"Parameter dimension: n={n}")
    print(f"Measurement dimension: m={m}")
    return y, fwd, x0, s

n1 = 60
n = n1*n1
print(n)
ensemblesize = 1500
id = np.identity(n)
test = EnsembleMode(covroot = id, options={"j1": ensemblesize})


# generate covariance approximation with straightforward sampling
ensemble = test._generate_ensemble()
a1 = test._anomaly(ensemble)
c1 = a1 @ a1.T
r1 = id - c1

# generate covariance approximation with new sampling technique
a2 = test._special_localization()
c2 = a2 @ a2.T
r2 = id - c2
plt.imshow(c2)
plt.show()

# compare this with an orthogonalized sampling technique
a3, r = np.linalg.qr(a1)
c3 = a3 @ a3.T
r3 = id - c3

# and with a deterministic approximation based on minimizing the j1-infinity error
a4 = test._deterministic()
c4 = a4 @ a4.T
r4 = id - c4

# tridiagonal
a5 = np.diag(np.random.randn(n)) + np.diag(np.random.randn(n-1), k=-1)
c5 = a5 @ a5.T
r5 = id - c5


fig, ax = plt.subplots(1,5)
ax[0].imshow(np.reshape(a1[:,20], (n1,n1)), cmap="gray")
ax[1].imshow(np.reshape(a2[:,20], (n1,n1)), cmap="gray")
ax[2].imshow(np.reshape(a3[:,20], (n1,n1)), cmap="gray")
ax[3].imshow(np.reshape(a4[:,20], (n1,n1)), cmap="gray")
ax[4].imshow(np.reshape(a5[:,20], (n1,n1)), cmap="gray")
plt.show()

def l1_vector(x):
    return np.sum(np.abs(x))

def foerstner(x):
    # compute eigenvalues
    s, u = eigh(x)
    s = s[s > 1e-10]    # 0 eigenvectors are a problem
    dist = np.sqrt(np.sum(np.log(s)**2))
    return dist

# evaluate some norms
print("NORM | CLASSICAL SAMPLING | NEW SAMPLING TECHNIQUE | ORTHOGONALIZED SAMPLING | j1-infinity minimization | tridiagonal")
print(f"l2 | {norm(r1)} | {norm(r2)} | {norm(r3)} | {norm(r4)} | {norm(r5)}")
print(f"l1 | {norm(r1, ord=1)} | {norm(r2, ord=1)} | {norm(r3, ord=1)} | {norm(r4, ord=1)} | {norm(r5, ord=1)}")
print(f"j1-infinity | {norm(r1, ord=np.inf)} | {norm(r2, ord=np.inf)} | {norm(r3, ord=np.inf)} | {norm(r4, ord=np.inf)} | {norm(r5, ord=np.inf)}")
print(f"Frobenius | {norm(r1, ord='fro')} | {norm(r2, ord='fro')} | {norm(r3, ord='fro')} | {norm(r4, ord='fro')} | {norm(r5, ord='fro')}")
print(f"Entrywise-l1 | {l1_vector(r1)} | {l1_vector(r2)} | {l1_vector(r3)} | {l1_vector(r4)} | {l1_vector(r5)}")
#print(f"Nuclear | {norm(r1, ord='nuc')} | {norm(r2, ord='nuc')} | {norm(r3, ord='nuc')} | {norm(r4, ord='nuc')}")
print(f"Entrywise j1-inf | {np.max(r1.flatten())} | {np.max(r2.flatten())} | {np.max(r3.flatten())} | {np.max(r4.flatten())} | {np.max(r5.flatten())}")
#print(f"j1 negative 2 | {norm(r1, ord=-2)} | {norm(r2, ord=-2)} | {norm(r3, ord=-2)} | {norm(r4, ord=-2)}")
print(f"Rank | {matrix_rank(r1)} | {matrix_rank(r2)} | {matrix_rank(r3)} | {matrix_rank(r4)} | {matrix_rank(r5)}")
print(f"Foerstner | {foerstner(c1)} | {foerstner(c2)} | {foerstner(c3)} | {foerstner(c4)} | {foerstner(r5)}")

