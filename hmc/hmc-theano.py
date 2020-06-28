"""HMC benchmark, Theano version."""
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

def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

def U(y, X, beta, alpha):
    return T.sum(T.log(1 + T.exp(T.dot(X, beta)))) - T.dot(T.transpose(y), (T.dot(X,beta))) + (0.5/alpha) * (T.dot(T.transpose(beta), beta))

def hmc(y, X, epsilon, L, start_q, alpha, n):
    ty = theano.shared(y)
    tX = theano.shared(X)

    def tU(beta):
        return U(ty, tX, beta, alpha)[0][0]

    def tgrad_U(beta):
        return T.grad(cost=tU(beta), wrt=beta)

    def update(current_q):
        q = current_q.copy()
        p = srng.normal(current_q.shape)
        current_p = p.copy()
        p = p - 0.5 * epsilon * tgrad_U(q)
        for i in range(L):
            # full step for the position
            q = q + epsilon * p
            # full step for the momentum, except at end of trajectory
            if i < L - 1:
                p = p - epsilon * tgrad_U(q)
            # half step for momentum at the end and negate to make the proposal symmetric
        p = -(p - 0.5 * epsilon * tgrad_U(q))
        current_U = tU(current_q)
        current_K = 0.5 * T.dot(T.transpose(current_p), current_p)
        proposed_U = tU(q)
        proposed_K = 0.5 * T.dot(T.transpose(p), p)
        ratio = (current_U - proposed_U + current_K - proposed_K)[0][0]
        return T.switch(T.lt(T.log(srng.uniform([])), ratio), q, current_q)

    t_start_q = theano.shared(start_q)
    result, updates = theano.scan(fn=update, outputs_info=t_start_q, n_steps=n)
    return np.squeeze(theano.function([], result, updates=updates)())

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
    out[f'theano-{args.mode}'] = t
    coef_ = np.mean(z[params['burn_in']:], 0)
    acc = np.mean((sigmoid(X_test@coef_) > 0.5) == np.squeeze(y_test))
    assert acc > 0.8
    print(json.dumps(out))