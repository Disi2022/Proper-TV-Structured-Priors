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
from pytensor.tensor.random.basic import uniform
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.distributions.dist_math import check_icdf_parameters, check_icdf_value
import scipy
import optuna

class BoundedContinuous(Continuous):
    """Base class for bounded continuous distributions."""

    # Indices of the arguments that define the lower and upper bounds of the distribution
    bound_args_indices: tuple[int | None, int | None] | None = None

GPU_ID = '0'
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
trial_number = 50

rs = 256
T1_gt, M_gt, mask, A1, A2, _, y_obs = load_data(rs)
A1_pts = pts.as_sparse_variable(A1)
A2_pts = pts.as_sparse_variable(A2)
mask_arr = mask != 0
N = np.sum(mask_arr)
s = (N,)


class TVUNI(BoundedContinuous):

    rv_op = uniform
    bound_args_indices = (2, 3)  # Lower, Upper

    @classmethod
    def dist(cls, lower, upper, **kwargs):
        lower = pt.as_tensor_variable(lower)
        upper = pt.as_tensor_variable(upper)
        return super().dist([lower, upper], **kwargs)

    def logp(value, lower, upper):
        res = - (pt.abs(pts.dot(A1_pts,value)) + pt.abs(pts.dot(A2_pts,value))) * lam_tv
        res = pt.switch(
            pt.bitwise_and(pt.ge(value, lower), pt.le(value, upper)),
            pt.fill(value, res),
            -np.inf,
        )

        return check_parameters(
            res,
            lower <= upper,
            msg="lower <= upper",
        )



waics = []
lam_tvs = []

def objective(trial):

    global lam_tv
    lam_tv = trial.suggest_float("lam_tv", 1e-3, 1)
 
    basic_model = pm.Model()
    with basic_model:
        T1 = TVUNI("T1", lower = 0, upper = 50, shape=s)
        M0 = TVUNI("M0", lower = 0, upper = 50, shape=s)

        # Likelihood (sampling distribution) of observations
        mu = VFA(T1, M0)

        sigma = pm.HalfNormal("sigma", sigma=0.01)
        
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y_obs[mask_arr])

        
        posterior = sample_numpyro_nuts(draws=2000, tune=2000, chains=4, target_accept=0.96, random_seed=123,idata_kwargs={"log_likelihood": True},
                                        chain_method='vectorized', compute_convergence_checks = False,
                                        progressbar = False,postprocessing_backend = 'cpu',
                                        ) 
           
    elpd_waic = az.waic(posterior).elpd_waic
    waics.append(elpd_waic)
    lam_tvs.append(lam_tv)

    if elpd_waic == max(waics):
        az.plot_trace(posterior,var_names=('T1'), coords={"T1_dim_0": 10})
        plt.savefig(f'figs/tvuni{trial.number}.png') 
        posterior.to_netcdf(f"results/tvuni.nc")

    if trial.number == trial_number-1:
        np.savez('results/waics_tvuni.npz',waics = waics,lam_tvl1s=lam_tvs)
    
    return elpd_waic


 
if __name__ == "__main__":
 
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trial_number)

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

