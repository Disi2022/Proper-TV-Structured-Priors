import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pymc.sampling.jax import sample_numpyro_nuts
import os
# -----------------------------
# Create a 2D “ground truth” image
# -----------------------------
GPU_ID = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


np.random.seed(42)
rs = 32  # image size
img_true = np.zeros((rs, rs))
img_true[8:16, 8:16] = 5
img_true[20:28, 20:28] = -3

# Mask some pixels (simulate missing data)
mask = np.ones_like(img_true, dtype=bool)
mask[0:4, :] = False
mask[:, 0:4] = False
mask_arr = mask.ravel()  # flatten mask

# Flattened observations
y_obs = img_true + np.random.normal(0, 1.0, size=img_true.shape)
y_vec = y_obs.ravel()[mask_arr]

# -----------------------------
# Gradient operators for TV
# -----------------------------
N = rs*rs
def build_gradients(rs):
    # Ax: difference along x (columns)
    Ax = np.zeros((N, N))
    for i in range(rs):
        for j in range(rs-1):
            idx = i*rs + j
            Ax[idx, idx] = -1
            Ax[idx, idx+1] = 1

    # Ay: difference along y (rows)
    Ay = np.zeros((N, N))
    for i in range(rs-1):
        for j in range(rs):
            idx = i*rs + j
            Ay[idx, idx] = -1
            Ay[idx, idx+rs] = 1

    return pt.as_tensor_variable(Ax), pt.as_tensor_variable(Ay)

Ax_t, Ay_t = build_gradients(rs)

# -----------------------------
# TV-L1 hyperparameters
# -----------------------------
lam_tv = 1.0
mu_l1 = 0.01
eps = 1e-6
def smooth_abs(x):
    return pt.sqrt(x**2 + eps)

# -----------------------------
# Build PyMC model
# -----------------------------
mask_idx = np.where(mask_arr)[0]

with pm.Model() as model:
    x = pm.DensityDist("x", logp=lambda value: pt.as_tensor_variable(0.0), shape=(N,))
    
    # TV-L1
    dx = Ax_t @ x
    dy = Ay_t @ x
    tv = smooth_abs(dx) + smooth_abs(dy)
    l1 = smooth_abs(x)
    pm.Potential("tv_l1", - (lam_tv * pt.sum(tv) + mu_l1 * pt.sum(l1)))

    # Likelihood
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    Y_obs = pm.Normal("Y_obs", mu=x[mask_idx], sigma=sigma, observed=y_vec)


    # Sample
    trace = sample_numpyro_nuts(
        draws=500,
        tune=500,
        chains=4,
        target_accept=0.96,
        random_seed=123,
        idata_kwargs={"log_likelihood": False},
        chain_method="vectorized",
        compute_convergence_checks=False,
        progressbar=True,
        postprocessing_backend="cpu",
    )

# -----------------------------
# Plot results
# -----------------------------
x_map = trace.posterior["x"].mean(dim=("chain","draw")).values
x_img = x_map.reshape(rs, rs)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Ground truth")
plt.imshow(img_true, cmap='coolwarm', vmin=-3, vmax=5)
plt.colorbar()
plt.subplot(1,3,2)
plt.title("Noisy observed")
plt.imshow(y_obs, cmap='coolwarm', vmin=-3, vmax=5)
plt.colorbar()
plt.subplot(1,3,3)
plt.title("TV–L1 reconstruction")
plt.imshow(x_img, cmap='coolwarm', vmin=-3, vmax=5)
plt.colorbar()
plt.savefig('results')
