"""HMC benchmark, Python/JAX version."""
import math
import random
import time
import json
import argparse
from typing import Callable
import numpy as np
import jax.numpy as jnp
import jax
from util import get_data


parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='std', choices=['std'])
args = parser.parse_args()


def hmc(y:jnp.ndarray, x:jnp.ndarray, epsilon:float, leapfrog_steps:int, alpha:float, n_iter:int) -> jnp.ndarray:
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
    @jax.jit
    def u(beta: jnp.array) -> jnp.array:
        x_beta = x@beta
        return jnp.sum(jax.nn.softplus(x_beta)) - y.T@(x_beta) + (0.5/alpha)*(beta.T@beta)
    
    # Define gradient of U (dU/dbeta)
    @jax.jit
    def d_u(beta: jnp.array) -> jnp.array:
        return x.T@(jax.nn.sigmoid(x@beta) - y) + beta/alpha
    
    @jax.jit
    def hmc_step(q_key:jnp.ndarray, _) -> jnp.ndarray:
        current_q, current_key = q_key
        key, next_key = jax.random.split(current_key)
        q = jnp.copy(current_q)
        
        # Sample momentum
        p = jax.random.normal(key, (len(current_q), 1))
        current_p = jnp.copy(p)
        
        # Half-step momentum update
        p -= 0.5*epsilon*d_u(q)
        
        # Leapfrog integrator
        for i in range(leapfrog_steps):
            q += epsilon*p
            # Momentum update (except last step)
            if i < leapfrog_steps - 1:
                p -= epsilon*d_u(q)
        
        # Final half-step momentum update (negated for symmetry)
        p = 0.5*epsilon*d_u(q) - p
        
        # Compute Hamiltonian values at current and proposed states
        current_u = u(current_q)
        current_k = 0.5*(current_p.T@current_p)
        proposed_u = u(q)
        proposed_k = 0.5*(p.T@p)
        ratio = (current_u - proposed_u + current_k - proposed_k)[0,0]
        
        # Acceptance
        out = jax.lax.select(jnp.log(jax.random.uniform(key)) < ratio, q, current_q)
        return (out, next_key), out

    initial_key = jax.random.key(0)
    # Initialize starting point
    initial_state = jnp.zeros((x.shape[1], 1), dtype=jnp.float32)
    _, samples = jax.lax.scan(hmc_step, (initial_state, initial_key), xs=None, length=n_iter)
    return np.array(samples.squeeze())

out = {}
# Load parameters from JSON file
    
with open('params.json', 'r') as params_file:
    params = json.load(params_file)

x_train, y_train, x_test, y_test = get_data()
y_train = jnp.array(np.expand_dims(y_train, 1))
x_train = jnp.array(x_train)
# Warm up run
hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], 1)
# Perform full HMC sampling
runtime = time.perf_counter()
samples = hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'])
runtime = time.perf_counter() - runtime
out['jax'] = runtime
# Estimate posterior mean after burn-in
posterior_mean = np.mean(samples[params['burn_in']:], 0)
# Predict on test set
sigmoid = lambda x: 1/(1 + np.exp(-x))
accuracy = np.mean((sigmoid(x_test@posterior_mean) > 0.5) == np.squeeze(y_test))
# Validate accuracy (must be > 80%)
assert accuracy > params['val_accuracy'], f"Accuracy too low: {accuracy}"
print(json.dumps(out))
