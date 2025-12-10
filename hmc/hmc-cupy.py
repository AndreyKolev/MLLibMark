"""HMC benchmark, python/CuPy version."""
import math
import random
from typing import Callable
import time
import json
import argparse
import numpy as np
import scipy.io
import sklearn.datasets as datasets
import cupy as cp
from cupyx.scipy.special import expit
from util import get_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='std', choices=['std'], help='')
args = parser.parse_args()
out = {}

# Numerically stable softplus: log(1 + exp(x)) 
def softplus(x: cp.ndarray) -> cp.ndarray:
    return cp.maximum(0, x) + cp.log1p(cp.exp(-cp.abs(x)))

def hmc_step(u:Callable, du:Callable, epsilon:float, leapfrog_steps:int, current_q:cp.ndarray) -> cp.ndarray:
    """Perform a single HMC step using the leapfrog integrator.
    
    Args:
        y: Target values (shape: [n_samples, 1]).
        x: Feature matrix (shape: [n_samples, n_features]).
        epsilon: Step size for the leapfrog integrator.
        leapfrog_steps: Number of leapfrog steps per HMC iteration.
        current_q: Current state (beta vector, shape: [n_features, 1]).
        alpha: Scale of the Gaussian prior.

    Returns:
        cupy.ndarray: New state (beta), either accepted or rejected.
    """
    q = current_q
    # Sample momentum
    p = cp.random.randn(1, len(current_q), dtype=cp.float32).T
    current_p = p
    # Half-step momentum update
    p = p - 0.5*epsilon*du(q)
    # Leapfrog integrator
    for i in range(leapfrog_steps):
        q += epsilon*p
        # Momentum update (except last step)
        if i < leapfrog_steps - 1:
            p -= epsilon*du(q)
    # Final half-step momentum update (negated for symmetry)
    p = 0.5*epsilon*du(q) - p
    # Compute potential and kinetic energies at current and proposed states
    current_u = u(current_q)
    current_k = 0.5*(current_p.T@current_p)
    proposed_u = u(q)
    proposed_k = 0.5*(p.T@p)
    # Compute acceptance probability
    if math.log(random.random()) < (current_u - proposed_u + current_k - proposed_k)[0]:
        return q
    return current_q

def hmc(y:cp.ndarray, x:cp.ndarray, epsilon:float, leapfrog_steps:int, alpha:float, n_iter:int) -> np.ndarray:
    """Run Hamiltonian Monte Carlo sampling for Bayesian logistic regression.

    This function performs HMC sampling to estimate the posterior distribution
    of the regression coefficients.

    Args:
        y: Training target values (shape: [n_samples, 1]).
        x: Training feature matrix (shape: [n_samples, n_features]).
        epsilon: Step size for leapfrog integrator.
        leapfrog_steps: Number of leapfrog steps per HMC iteration.
        alpha: regularization strength.
        n_iter: Total number of HMC iterations to run.

    Returns:
        np.ndarray: posterior samples (shape: [n_iter, n_features]).
    """
    # Define negative log-posterior (potential energy)  
    def u(beta:cp.ndarray):
        x_beta = x@beta
        return cp.sum(softplus(x_beta)) - y.T@(x_beta) + (0.5/alpha)*cp.sum(cp.square(beta))
    # Define gradient of U (dU/dbeta)
    def du(beta):
        return x.T@(expit(x@beta) - y) + beta/alpha
 
    q = cp.zeros((x.shape[1], 1), dtype=cp.float32)
    out = cp.zeros((n_iter, x.shape[1]), dtype=cp.float32)
    for i in range(n_iter):
        q = hmc_step(u, du, epsilon, leapfrog_steps, q)
        out[i] = q.ravel()
    return cp.asnumpy(out)

out = {}
# Load hyper params
with open('params.json') as params_file:
    params = json.load(params_file)
# Load dataset
x_train, y_train, x_test, y_test = get_data()
y_train = cp.array(np.expand_dims(y_train, 1), dtype=cp.float32)
x_train = cp.array(x_train, dtype=cp.float32)
# Warm up run
hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], 1)  # Warm-up
# Perform full HMC sampling
runtime = time.perf_counter()
samples = hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'])  
runtime = time.perf_counter() - runtime
out[f'cupy'] = runtime
# Estimate posterior mean after burn-in
posterior_mean = np.mean(samples[params['burn_in']:], 0)
# Predict on test set
sigmoid = lambda x: 1.0/(1.0 + np.exp(-x))
accuracy = np.mean((sigmoid(x_test@posterior_mean) > 0.5) == np.squeeze(y_test))
assert accuracy > params['val_accuracy'], f"Accuracy too low: {accuracy}"
# Output result
print(json.dumps(out))
