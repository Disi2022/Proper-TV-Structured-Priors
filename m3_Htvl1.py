import numpy as np
from utils import load_data,VFA
import arviz as az
import pytensor.tensor as pt
import pytensor.sparse as pts
from pytensor.tensor import TensorVariable
from pymc.distributions.dist_math import check_parameters
import pymc as pm
from pymc.sampling.jax import sample_numpyro_nuts
import matplotlib.pyplot as plt
import os
from pymc.distributions.distribution import Continuous
from pytensor.tensor.random.basic import laplace
import scipy

GPU_ID = '0'
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

rs = 256
T1_gt, M_gt, mask, A1,A2,A3, y_obs = load_data(rs)
A1_pts = pts.as_sparse_variable(A1)
A2_pts = pts.as_sparse_variable(A2)
A3_pts = pts.as_sparse_variable(A3)
mask_arr = mask != 0
N = np.sum(mask_arr)
s = (N,)

class TVL1(Continuous):

    rv_op = laplace
    
    @classmethod
    def dist(cls, mu, b, *args, **kwargs):
        b = pt.as_tensor_variable(b)
        mu = pt.as_tensor_variable(mu)
        return super().dist([4, 1], *args, **kwargs)

    def logp(value,mu,b):
        res = - (pt.abs(pts.dot(A1_pts,value))  + pt.abs(pts.dot(A2_pts,value))) * mu - pt.abs(pts.dot(A3_pts,value)) * b  #- pt.log(2 * b)

        return res


# ---------- Model ----------
with pm.Model() as basic_model:
    # Hyperpriors
    lam = pm.Exponential('lam',lam=1)
    mul1 = pm.Exponential('mul1', lam=1)     

    T1 = TVL1('T1', mu = lam, b = mul1, shape=s)
    M = TVL1('M', mu = lam, b = mul1, shape=s)

    # Forward model
    mu = VFA(T1, M)

    # Observation noise
    sigma = pm.HalfNormal("sigma", sigma=0.4)

    # Likelihood
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y_obs[mask_arr])

    # Sampling
    posterior = sample_numpyro_nuts(draws=2000, tune=2000, chains=4, target_accept=0.96, random_seed=123,idata_kwargs={"log_likelihood": False},
                                    chain_method='vectorized',progressbar = True,postprocessing_backend = 'cpu',
                                    ) 

    # Save output
    posterior.to_netcdf(f"results/Htvl1.nc")

    # Plot
    az.plot_trace(posterior, var_names=('T1'), coords={"T1_dim_0": 10})
    plt.savefig(f'figs/htvl1.png')