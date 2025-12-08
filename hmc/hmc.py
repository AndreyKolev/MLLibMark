"""HMC benchmark, numpy version."""
import math
import random
import time
import json
import argparse
from typing import Callable
import numpy as np
from util import get_data


parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='std', choices=['std'])
args = parser.parse_args()

# Sigmoid function
def sigmoid(x:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))

# Numerically stable log(1 + exp(x)) 
def log1p_exp(x: np.ndarray) -> np.ndarray:
    return x*(x > 0) + np.log1p(np.exp(-np.abs(x)))


def hmc_step(u:Callable, d_u:Callable, epsilon:float, leapfrog_steps:int, current_q:np.ndarray) -> np.ndarray:
    """Perform a single HMC step using the leapfrog integrator.
    
    Args:
        u: Potential energy function.
        d_u: Gradient of potential energy function u.
        epsilon: Step size for the leapfrog integrator.
        leapfrog_steps: Number of leapfrog steps per HMC iteration.
        current_q: Current state (beta vector, shape: [n_features, 1]).
        alpha: Scale of the Gaussian prior.

    Returns:
        New state (beta), either accepted or rejected.
    """
    q = np.copy(current_q)
    rng = np.random.default_rng()
    # Sample momentum
    p = rng.standard_normal(size=(len(current_q), 1), dtype=np.float32)
    current_p = np.copy(p)
    # Half-step momentum update
    p = p - 0.5*epsilon*d_u(q)
    
    # Leapfrog integrator
    for i in range(leapfrog_steps):
        q = q + epsilon*p
        # Momentum update (except last step)
        if i < leapfrog_steps - 1:
            p = p - epsilon*d_u(q)
    
    # Final half-step momentum update (negated for symmetry)
    p = 0.5*epsilon*d_u(q) - p
    # Compute Hamiltonian values at current and proposed states
    current_u = u(current_q)
    current_k = 0.5*(current_p.T@current_p)
    proposed_u = u(q)
    proposed_k = 0.5*(p.T@p)
    # Acceptance
    if math.log(random.random()) < (current_u - proposed_u + current_k - proposed_k).item():
        return q  # Accept proposal
    return current_q  # Reject and return current state


def hmc(y:np.ndarray, x:np.ndarray, epsilon:float, leapfrog_steps:int, alpha:float, n_iter:int) -> np.ndarray:
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
        Numpy array of posterior samples (shape: [n_iter, n_features]).
    """
    # Define negative log-posterior (potential energy)  
    def u(beta: np.array) -> np.array:
        x_beta = x@beta
        return np.sum(log1p_exp(x_beta)) - y.T@(x_beta) + (0.5/alpha)*np.sum(np.square(beta))
    # Define gradient of U (dU/dbeta)
    def d_u(beta: np.array) -> np.array:
        return x.T@(sigmoid(x@beta) - y) + beta/alpha

    dim = x.shape[1]
    # Initialize starting point
    state = np.zeros((dim, 1), dtype=np.float32)
    # Storage for results
    samples = np.zeros((n_iter, dim), dtype=np.float32)
    # Run HMC iterations
    for i in range(n_iter):
        state = hmc_step(u, d_u, epsilon, leapfrog_steps, state)
        samples[i] = state.ravel()
    return samples


with open('params.json', 'r') as params_file:
    out = {}
    # Load parameters from JSON file
    params = json.load(params_file)
    x_train, y_train, x_test, y_test = get_data()
    y_train = np.expand_dims(y_train, 1)
    # Warm up run
    z = hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], 1)  # Warm-up
    # Perform full HMC sampling
    t = time.perf_counter()
    z = hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'])
    t = time.perf_counter() - t
    out['numpy-std'] = t
    # Estimate posterior mean after burn-in
    posterior_mean = np.mean(z[params['burn_in']:], 0)
    # Predict on test set
    accuracy = np.mean((sigmoid(x_test@posterior_mean) > 0.5) == np.squeeze(y_test))
    # Validate accuracy (must be > 80%)
    assert accuracy > params['val_accuracy'], f"Accuracy too low: {accuracy}"
    print(json.dumps(out))
