"""HMC benchmark, Tensorflow V1 version."""
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

def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))
    
def U(y, X, beta, alpha):
    return tf.reduce_sum(tf.log(1 + tf.exp(X@beta))) - tf.transpose(y)@(X@beta) + (0.5/alpha) * (tf.transpose(beta)@beta)

def hmc(y, X, epsilon, L, start_q, alpha, n):
    config = tf.ConfigProto()
    if args.mode == 'cpu':
        config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        ty = tf.Variable(initial_value=y)
        tX = tf.Variable(initial_value=X)
        tU = lambda beta: U(ty, tX, beta, alpha)
        tgrad_U = lambda beta: tf.gradients(ys=U(ty, tX, beta, alpha), xs=beta)[0] #lambda beta: grad_U(ty, tX, beta, alpha)
        z = tf.Variable(initial_value=np.zeros(n, dtype='float32'))

        cur_q = tf.Variable(initial_value=start_q)
        sess.run(tf.global_variables_initializer())
        
        def update(current_q, _):
            q = tf.identity(current_q)
            p = tf.random.normal(current_q.get_shape()) 
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
            return tf.cond(tf.less(tf.log(tf.random.uniform(())), ratio), lambda: q, lambda: current_q)

        sc = tf.squeeze(tf.scan(update, z, initializer=cur_q))
        return sess.run(sc)

with open('params.json') as params_file:
    out = {}
    params = json.load(params_file)
    X_train, y_train, X_test, y_test = get_data()
    y_train = np.expand_dims(y_train, 1)
    D = X_train.shape[1]
    q = np.zeros((D, 1), dtype='float32')
    z = hmc(y_train, X_train, params['epsilon'], params['n_leaps'], q, params['alpha'], 1)  # Warm-up
    t = time.perf_counter()
    z = hmc(y_train, X_train, params['epsilon'], params['n_leaps'], q, params['alpha'], params['n_iter'])  
    t = time.perf_counter() - t
    out[f'tensorflow-{args.mode}'] = t
    coef_ = np.mean(z[params['burn_in']:], 0)
    acc = np.mean((sigmoid(X_test@coef_) > 0.5) == np.squeeze(y_test))
    assert acc > 0.8
    print(json.dumps(out))