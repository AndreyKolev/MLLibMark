"""HMC benchmark, Python/Theano version."""
import numpy as np
import math
import time
import os
import argparse
import json
from util import get_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
out = {}

device = 'cuda' if args.mode == 'gpu' else 'cpu'
flags = f'mode=FAST_RUN, device={device}, floatX=float32'
os.environ['THEANO_FLAGS'] = flags

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed=0)

# Define negative log-posterior (potential energy)  
def u(y, x, beta, alpha):
    x_beta = T.dot(x, beta)
    return T.sum(T.nnet.softplus(x_beta)) - T.dot(y.T, x_beta) + (0.5/alpha)*T.dot(beta.T, beta)

# Gradient of the energy function U with respect to beta
def grad_u(y, x, beta, alpha):
    return T.dot(x.T, T.nnet.sigmoid(T.dot(x, beta)) - y) + beta/alpha


def hmc(y:np.ndarray, x:np.ndarray, epsilon:float, leapfrog_steps:int, alpha:float, n:int):
    """Run Hamiltonian Monte Carlo sampling for Bayesian logistic regression.

    This function performs HMC sampling to estimate the posterior distribution
    of the regression coefficients.

    Args:
        y: Training target values (shape: [n_samples, 1]).
        x: Training feature matrix (shape: [n_samples, n_features]).
        epsilon: Step size for leapfrog integrator.
        leapfrog_steps: Number of leapfrog steps per HMC iteration.
        alpha: regularization strength.
        n: Total number of HMC iterations to run.

    Returns:
        Numpy array of posterior samples (shape: [n_iter, n_features]).
    """
    ty = theano.shared(y)
    tx = theano.shared(x)
    
    def tu(beta):
        return u(ty, tx, beta, alpha)[0,0]

    def tgrad_u(beta):
        #T.grad(cost=tu(beta), wrt=beta)
        return grad_u(ty, tx, beta, alpha)
    
    # HMC sampler (single step)
    def hmc_step(current_q):
        q = current_q.copy()
        p = srng.normal(current_q.shape)
        current_p = p.copy()
        p = p - 0.5*epsilon*tgrad_u(q)
        for i in range(leapfrog_steps):
            # full step for the position
            q = q + epsilon * p
            # full step for the momentum, except at end of trajectory
            if i < leapfrog_steps - 1:
                p = p - epsilon*tgrad_u(q)
        # Final half-step momentum update (negated for symmetry)
        p = 0.5*epsilon*tgrad_u(q) - p
        # Compute potential and kinetic energies at current and proposed states
        current_u = tu(current_q)
        current_k = 0.5*T.dot(current_p.T, current_p)
        proposed_u = tu(q)
        proposed_k = 0.5*T.dot(p.T, p)
        # Compute acceptance probability
        ratio = (current_u - proposed_u + current_k - proposed_k)[0,0]
        return T.switch(T.lt(T.log(srng.uniform()), ratio), q, current_q)
    
    # Initialize starting point
    start_q = theano.shared(np.zeros((x.shape[1], 1), dtype=np.float32))
    # Run HMC iterations
    result, samples = theano.scan(fn=hmc_step, outputs_info=start_q, n_steps=n)
    return np.squeeze(theano.function([], result, updates=samples)())

out = {}
# Load hyperparams from JSON file
with open('params.json') as params_file:
    params = json.load(params_file)

x_train, y_train, x_test, y_test = get_data()
y_train = np.expand_dims(y_train, 1)

# Warm up run
hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], 1)  # Warm-up

# Perform full HMC sampling
runtime = time.perf_counter()
samples = hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'])  
runtime = time.perf_counter() - runtime
out[f'theano-{args.mode}'] = runtime

# Estimate posterior mean after burn-in
posterior_mean = np.mean(samples[params['burn_in']:], 0)

# Predict on test set
sigmoid = lambda x: 1.0/(1.0 + np.exp(-x))
accuracy = np.mean((sigmoid(x_test@posterior_mean) > 0.5) == y_test.squeeze())

# Validate accuracy
assert accuracy > params['val_accuracy'], f"Accuracy too low: {accuracy}"
# Output result
print(json.dumps(out))
