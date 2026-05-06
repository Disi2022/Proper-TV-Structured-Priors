


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
import pickle


GPU_ID = '0'
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

rs = 256
T1_gt, M_gt, mask, _,_,_, y_obs = load_data(rs)
mask_arr = mask != 0
N = np.sum(mask_arr)
s = (N,)


basic_model = pm.Model()
with basic_model:
    
    T1 = pm.Uniform('T1', lower=0,upper=50, shape = s)
    M = pm.Uniform('M', lower=0,upper=50, shape = s)

    # Likelihood (sampling distribution) of observations
    mu = VFA(T1, M)

    sigma = pm.HalfNormal("sigma", sigma=0.4)
    
    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y_obs[mask_arr,:])

    
    posterior = sample_numpyro_nuts(draws=2000, tune=2000, chains=4, target_accept=0.96, random_seed=123,idata_kwargs={"log_likelihood": False},
                                    chain_method='sequential', compute_convergence_checks = False,
                                    progressbar = True,postprocessing_backend = 'cpu',
                                    ) 

    mp = pm.find_MAP()
    T1 = mp['T1']

    with open('results/map_estimates.pkl', 'wb') as f:
        pickle.dump({'T1': T1}, f)
    
    posterior.to_netcdf(f"results/uni.nc")
    az.plot_trace(posterior,var_names=('T1'), coords={"T1_dim_0": 10})
    plt.savefig(f'figs/uni.png')

