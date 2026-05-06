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
import optuna
from optuna.trial import TrialState

GPU_ID = '0'
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

rs = 256
T1_gt, M_gt, mask, A1,A2,A3, y_obs = load_data(rs)
A = scipy.sparse.vstack([A1, A2], format='csr')  # Ensures efficient storage
A1_pts = pts.as_sparse_variable(A1)
A2_pts = pts.as_sparse_variable(A2)
A3_pts = pts.as_sparse_variable(A3)
A_pts = pts.as_sparse_variable(A)
mask_arr = mask != 0
N = np.sum(mask_arr)
s = (N,)

waics = []
lam_tvl1s = []
mu_tvl1s = []
class TVL1(Continuous):

    rv_op = laplace

    @classmethod
    def dist(cls, mu, b, *args, **kwargs):
        b = pt.as_tensor_variable(b)
        mu = pt.as_tensor_variable(mu)
        return super().dist([4, 1], *args, **kwargs)

    def logp(value,mu,b):
        res = - (pt.abs(pts.dot(A1_pts,value)) + pt.abs(pts.dot(A2_pts,value))) * mu - pt.abs(pts.dot(A3_pts,value)) * b #- pt.log(2 * b)

        return res

def objective(trial):
    global lam_tvl1, mu_tvl1

    lam_tvl1 = trial.suggest_float("lam_tvl1", 1e-4, 10)
    mu_tvl1 = trial.suggest_float("mu_tvl1", 1e-4, 10)

    basic_model = pm.Model()
    with basic_model:
  
        T1 = TVL1("T1", mu=lam_tvl1, b =mu_tvl1, shape=s)
        M = TVL1("M", mu=lam_tvl1, b = mu_tvl1, shape=s)

        mu = VFA(T1, M)

        sigma = pm.HalfNormal("sigma", sigma=0.4)
        
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y_obs[mask_arr])

        
        posterior = sample_numpyro_nuts(draws=2000, tune=2000, chains=4, target_accept=0.96, random_seed=123,idata_kwargs={"log_likelihood": True},
                                        chain_method='vectorized', compute_convergence_checks = False,
                                        progressbar = False,postprocessing_backend = 'cpu',
                                        ) 
          
    elpd_waic = az.waic(posterior).elpd_waic
    waics.append(elpd_waic)
    lam_tvl1s.append(lam_tvl1)
    mu_tvl1s.append(mu_tvl1)

    if elpd_waic == max(waics):
        posterior.to_netcdf(f"results/tvl1.nc")
        az.plot_trace(posterior,var_names=('T1'), coords={"T1_dim_0": 10})
        plt.savefig(f'figs/tvl1_{trial.number}.png')  
    
    if trial.number == trial_number-1:
        np.savez('results/waics_tvl1.npz',waics = waics,lam_tvl1s=lam_tvl1s,mu_tvl1s=mu_tvl1s )
    
    return elpd_waic


 
if __name__ == "__main__":
 
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
