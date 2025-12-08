"""HMC benchmark, Tensorflow v2 version."""
import tensorflow as tf
import numpy as np
import math
import time
import json
from util import get_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()

# Set device
if args.mode == 'cpu':
    tf.config.set_visible_devices([], 'GPU')  # Use CPU


# Define negative log-posterior (potential energy)  
@tf.function
def u(y:tf.Tensor, x:tf.Tensor, beta:tf.Tensor, alpha:float):
    return tf.reduce_sum(tf.math.softplus(x@beta)) - tf.transpose(y)@(x@beta) + (0.5/alpha)*(tf.transpose(beta)@beta)

    
# Define gradient of U (dU/dbeta)
@tf.function
def grad_u(y:tf.Tensor, x:tf.Tensor, beta:tf.Tensor, alpha:float):
    return tf.transpose(x)@(tf.sigmoid(x@beta) - y) + beta/alpha


# Hamiltonian Monte Carlo (HMC) Sampler    
@tf.function
def hmc(y:tf.Tensor, x:tf.Tensor, epsilon:float, leapfrog_steps:int, alpha:float, n:int) -> tf.Tensor:
    """
    Hamiltonian Monte Carlo (HMC) sampler for logistic regression.

    Args:
        y (tf.Tensor): Training labels of shape (n_samples, 1).
        x (tf.Tensor): Training feature matrix of shape (n_samples, n_features).
        epsilon (float): Step size for the leapfrog integrator.
        leapfrog_steps (int): Number of leapfrog steps per HMC iteration.
        alpha (float): regularization strength. 
        n (int): Number of HMC samples to generate.

    Returns:
        tf.Tensor: Array of shape (n, n_features) containing sampled parameter vectors.
    """
    samples = tf.zeros(n)
    cur_q = tf.zeros((x.shape[1], 1))
    
    def hmc_step(current_q:tf.Tensor, _) -> tf.Tensor:
        q = tf.identity(current_q)
        p = tf.random.normal(current_q.shape) 
        current_p = tf.identity(p)
        p -= 0.5*epsilon*grad_u(y, x, q, alpha)
        # Leapfrog integrator
        for i in range(leapfrog_steps):
            # Position update
            q += epsilon*p
            # Momentum update (except last step)
            if i < leapfrog_steps - 1:
                p -= epsilon*grad_u(y, x, q, alpha)
        # Final half-step momentum update (negated for symmetry)
        p = 0.5*epsilon*grad_u(y, x, q, alpha) - p
        # Compute Hamiltonian values at current and proposed states
        current_u = u(y, x, current_q, alpha)
        current_k = 0.5*(tf.transpose(current_p)@current_p)
        proposed_u = u(y, x, q, alpha)
        proposed_k = 0.5*(tf.transpose(p)@p)
        # Compute log acceptance ratio
        ratio = (current_u - proposed_u + current_k - proposed_k)[0,0]
        # Accept or reject the proposed state
        if tf.math.log(tf.random.uniform(())) < ratio:
            return q
        return current_q

    return tf.squeeze(tf.scan(hmc_step, samples, initializer=cur_q))

out = {}
# Load hyper parameters from JSON file
with open('params.json') as params_file:
    params = json.load(params_file)

x_train, y_train, x_test, y_test = get_data()
y_train = np.expand_dims(y_train, 1)

# Warm up run
hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], 1) 

# Perform full HMC sampling
runtime = time.perf_counter()
samples = hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter']).numpy() 
runtime = time.perf_counter() - runtime

# Estimate posterior mean after burn-in
posterior_mean = np.mean(samples[params['burn_in']:], 0)

# Predict on test set
sigmoid = lambda x: 1/(1 + np.exp(-x))
accuracy = np.mean((sigmoid(x_test@posterior_mean) > 0.5) == np.squeeze(y_test))

# Validate accuracy
assert accuracy > params['val_accuracy'], f"Accuracy too low: {accuracy}"

out[f'tensorflow-{args.mode}'] = runtime
print(json.dumps(out))
