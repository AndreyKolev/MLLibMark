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
if args.mode == 'cpu':
    tf.config.set_visible_devices([], 'GPU')  # Use CPU

def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

def U(y, X, beta, alpha):
    return tf.reduce_sum(tf.math.log(1+tf.exp(X@beta))) -\
           tf.transpose(y)@(X@beta) + (0.5/alpha) * (tf.transpose(beta)@beta)

def grad_U(y, X, beta, alpha):
    return tf.transpose(X)@(tf.exp(X@beta)/(1+tf.exp(X@beta)) - y) + beta/alpha
    
@tf.function
def hmc(y, X, epsilon, L, start_q, alpha, n):
    tU = lambda beta: U(y, X, beta, alpha)
    tgrad_U = lambda beta: grad_U(y, X, beta, alpha)
    z = tf.zeros(n, tf.float32)
    cur_q = start_q
    
    def update(current_q, _):
        q = tf.identity(current_q)
        p = tf.random.normal(current_q.shape) 
        current_p = tf.identity(p)
        p = p - 0.5*epsilon*tgrad_U(q)
        for i in range(L):
            # position step
            q = q + epsilon*p
            # momentum step
            if i < L-1:
                p = p - epsilon*tgrad_U(q)
        # negate for symmetry
        p = -(p - 0.5*epsilon*tgrad_U(q))
        current_U = tU(current_q)
        current_K = 0.5*(tf.transpose(current_p)@current_p)
        proposed_U = tU(q)
        proposed_K = 0.5*(tf.transpose(p)@p)
        ratio = (current_U - proposed_U + current_K - proposed_K)[0][0]
        if tf.math.log(tf.random.uniform(())) < ratio:
            return q
        return current_q

    return tf.squeeze(tf.scan(update, z, initializer=cur_q))

with open('params.json') as params_file:
    out = {}
    params = json.load(params_file)
    X_train, y_train, X_test, y_test = get_data()
    y_train = np.expand_dims(y_train, 1)
    D = X_train.shape[1]
    q = np.zeros((D, 1), dtype='float32')
    z = hmc(y_train, X_train, params['epsilon'], params['n_leaps'], q,
            params['alpha'], 1)  # Warm-up
    t = time.perf_counter()
    z = hmc(y_train, X_train, params['epsilon'], params['n_leaps'], q,
            params['alpha'], params['n_iter']).numpy() 
    t = time.perf_counter()-t
    out[f'tensorflow-{args.mode}'] = t
    coef_ = np.mean(z[params['burn_in']:], 0)
    acc = np.mean((sigmoid(X_test@coef_) > 0.5) == np.squeeze(y_test))
    assert acc > 0.8
    print(json.dumps(out))