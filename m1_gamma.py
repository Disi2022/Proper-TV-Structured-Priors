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

GPU_ID = '0'
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

rs = 256
T1_gt, M_gt, mask, A1,A2,A3, y_obs = load_data(rs)
mask_arr = mask != 0
N = np.sum(mask_arr)
s = (N,)

basic_model = pm.Model()
with basic_model:
    
    T1 = pm.Gamma('T1', alpha=3, beta=1, shape = s)
    M = pm.Gamma('M', alpha=3, beta=1, shape = s)

    mu = VFA(T1, M)

    sigma = pm.HalfNormal("sigma", sigma=0.4)
    
    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y_obs[mask_arr])

    
    posterior = sample_numpyro_nuts(draws=2000, tune=2000, chains=4, target_accept=0.96, random_seed=123,idata_kwargs={"log_likelihood": False},
                                    chain_method='vectorized', compute_convergence_checks = True,
                                    progressbar = True,postprocessing_backend = 'cpu',
                                    ) 
    
    
    posterior.to_netcdf(f"results/gamma.nc")
    az.plot_trace(posterior,var_names=('T1'), coords={"T1_dim_0": 10})
    plt.savefig(f'figs/gamma.png')
