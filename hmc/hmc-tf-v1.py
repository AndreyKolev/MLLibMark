"""HMC benchmark, Tensorflow v1 version."""
import numpy as np
import math
import time
import json
from util import get_data
import argparse
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
    
# Define negative log-posterior (potential energy)  
def u(y, x, beta, alpha):
    return tf.reduce_sum(tf.math.softplus(x@beta)) - tf.transpose(y)@(x@beta) + (0.5/alpha)*(tf.transpose(beta)@beta)

# Define gradient of the potential energy U with respect to beta
def grad_u(y, x, beta, alpha):
    return tf.transpose(x)@(tf.sigmoid(x@beta) - y) + beta/alpha

def hmc(y:np.ndarray, x:np.ndarray, epsilon:float, leapfrog_steps:int, alpha:float, n:int) -> np.ndarray:
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
    config = tf.ConfigProto()
    if args.mode == 'cpu':
        config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        start_q = tf.zeros((x.shape[1], 1))
        ty = tf.Variable(initial_value=y)
        tx = tf.Variable(initial_value=x)
        tu = lambda beta: u(ty, tx, beta, alpha)
        tgrad_u = lambda beta: grad_u(ty, tx, beta, alpha)
        samples = tf.zeros(n)
        
        cur_q = tf.Variable(initial_value=start_q)
        sess.run(tf.global_variables_initializer())
        
        def update(current_q, _):
            q = tf.identity(current_q)
            p = tf.random.normal(current_q.get_shape()) 
            current_p = tf.identity(p)
            p = p - 0.5*epsilon*tgrad_u(q)
            # Leapfrog integrator
            for i in range(leapfrog_steps):
                # Position update
                q = q + epsilon*p
                 # Momentum update (except last step)
                if i < leapfrog_steps - 1:
                    p = p - epsilon*tgrad_u(q)
            # Final half-step momentum update (negated for symmetry)
            p = 0.5*epsilon*tgrad_u(q) - p
            current_u = tu(current_q)
            current_k = 0.5*(tf.transpose(current_p)@current_p)
            proposed_u = tu(q)
            proposed_k = 0.5*(tf.transpose(p)@p)
            # Compute acceptance ratio
            ratio = (current_u - proposed_u + current_k - proposed_k)[0,0]
            # Accept or reject the proposed state
            return tf.cond(tf.less(tf.log(tf.random.uniform(())), ratio), lambda: q, lambda: current_q)

        sc = tf.squeeze(tf.scan(update, samples, initializer=cur_q))
        return sess.run(sc)

out = {}
# Load hyper parameters from JSON file
with open('params.json') as params_file:
    params = json.load(params_file)
X_train, y_train, X_test, y_test = get_data()
y_train = np.expand_dims(y_train, 1)

# Warm up run
hmc(y_train, X_train, params['epsilon'], params['n_leaps'], params['alpha'], 1)  # Warm-up

# Perform full HMC sampling
runtime = time.perf_counter()
samples = hmc(y_train, X_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'])  
runtime = time.perf_counter() - runtime
out[f'tensorflow-v1-{args.mode}'] = runtime

# Find posterior mean after burn-in
posterior_mean = np.mean(samples[params['burn_in']:], 0)

# Predict on test set
sigmoid = lambda x: 1/(1 + np.exp(-x))
accuracy = np.mean((sigmoid(X_test@posterior_mean) > 0.5) == np.squeeze(y_test))

# Validate accuracy
assert accuracy > params['val_accuracy'], f"Accuracy too low: {accuracy}"
print(json.dumps(out))
