"""HMC benchmark, Python/mxnet version."""
import numpy as np
import math
import random
import time
import json
import argparse
from mxnet import nd, cpu, gpu
from util import get_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='Execution mode: use "cpu" for CPU-only computation or "gpu" for GPU-accelerated execution (if available).')
args = parser.parse_args()
out = {}

# Numerically stable softplus: log(1 + exp(x)) 
def softplus(x: nd.NDArray) -> nd.NDArray:
    return nd.relu(x) + nd.log1p(nd.exp(-nd.abs(x)))


# Define negative log-posterior (potential energy)  
def u(y:nd.NDArray, x:nd.NDArray, beta:nd.NDArray, alpha:float) -> nd.NDArray:
    x_beta = nd.dot(x, beta)
    return softplus(x_beta).sum() - nd.dot(y.T, x_beta) + (0.5/alpha)*nd.dot(beta.T, beta)


# Define gradient of U (dU/dbeta)
def grad_u(y:nd.NDArray, x:nd.NDArray, beta:nd.NDArray, alpha:float) -> nd.NDArray:
    return nd.dot(x.T, nd.sigmoid(nd.dot(x, beta)) - y) + beta/alpha


def hmc_step(y, x, epsilon, leapfrog_steps, current_q, alpha) -> nd.NDArray:
    """Perform a single HMC step using the leapfrog integrator.
    
    Args:
        u: Potential energy function.
        d_u: Gradient of potential energy function u.
        epsilon: Step size for the leapfrog integrator.
        leapfrog_steps: Number of leapfrog steps per HMC iteration.
        current_q: Current state (beta vector, shape: [n_features, 1]).

    Returns:
        mxnet.nd.NDArray: New state (beta), either accepted or rejected.
    """
    q = current_q
    p = nd.random.randn(len(current_q), 1, dtype='float32')
    current_p = p
    # Half-step momentum update
    p = p - 0.5*epsilon*grad_u(y, x, q, alpha)
    # Leapfrog integrator
    for i in range(leapfrog_steps):
        # Position update
        q += epsilon*p
        # Momentum update (except last step)
        if i < leapfrog_steps - 1:
            p -= epsilon*grad_u(y, x, q, alpha)
    
    # Final half-step momentum update (negated for symmetry)
    p = 0.5*epsilon*grad_u(y, x, q, alpha) - p
    
    # Compute potential and kinetic energies at current and proposed states
    current_u = u(y, x, current_q, alpha)
    current_k = 0.5*nd.dot(current_p.T, current_p)
    proposed_u = u(y, x, q, alpha)
    proposed_k = 0.5*nd.dot(p.T, p)

    # Acceptance
    if math.log(random.random()) < (current_u - proposed_u + current_k - proposed_k):
        return q  # Accept proposal
    return current_q  # Reject and return current state


def hmc(y:nd.NDArray, x:nd.NDArray, epsilon:float, leapfrog_steps:int, alpha:float, n_iter:int) -> nd.NDArray:
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
        mxnet.nd.NDArray: array of posterior samples (shape: [n_iter, n_features]).
    """
    #def u(beta):
    #    return nd.sum(nd.log(1 + nd.exp(nd.dot(X, beta)))) - nd.dot(y.T,(nd.dot(X,beta))) + (0.5/alpha)*nd.sum(beta**2)

    #def du(beta):
    #    return nd.dot(X.T, (nd.exp(nd.dot(X,beta))/(1+nd.exp(nd.dot(X,beta))) - y)) + beta/alpha

    q = nd.zeros((x.shape[1], 1), dtype='float32')
    out = nd.zeros((n_iter, x.shape[1]), dtype='float32')
    for i in range(n_iter):
        q = hmc_step(y, x, epsilon, leapfrog_steps, q, alpha)
        out[i] = q.squeeze()
    return out

out = {}
# Load hyper params
with open('params.json') as params_file:
    params = json.load(params_file)
# Load data & convert to ndarrays
x_train, y_train, x_test, y_test = get_data()

with cpu() if args.mode == 'cpu' else gpu(0):
    x_train = nd.array(x_train)
    y_train = nd.expand_dims(nd.array(y_train), 1)
    x_test = nd.array(x_test)
    y_test = nd.array(y_test)
    # Warm up run

    hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], 1)  # Warm-up
    # Perform full HMC sampling
    runtime = time.perf_counter()
    samples = hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'])  
    runtime = time.perf_counter() - runtime
    out[f'mxnet-{args.mode}'] = runtime
    posterior_mean = nd.mean(samples[params['burn_in']:], 0)

    # Predict on test set
    #sigmoid = lambda x: 1.0/(1.0 + np.exp(-x))
    accuracy = nd.mean((nd.sigmoid(nd.dot(x_test, posterior_mean)) > 0.5) == y_test)

# Validate accuracy
assert accuracy > params['val_accuracy'], f"Accuracy too low: {accuracy}"

# Output result
print(json.dumps(out))
