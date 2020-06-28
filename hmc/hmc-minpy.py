"""HMC benchmark, MinPy version."""
import numpy as np
import math
import random
import time
import json
import argparse
from minpy.context import cpu, gpu
import minpy.numpy as mp
from util import get_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
out = {}

def sigmoid(x):
    return 1.0/(1.0 + mp.exp(-x))

def hmc(U, dU, epsilon, L, current_q):
    q = current_q
    p = mp.random.randn(1, current_q.shape[0], dtype=mp.float32).T
    current_p = p
    # half step for momentum
    p = p - 0.5*epsilon*dU(q)
    # full steps for pos and momentum
    for i in range(L):
        q = q + epsilon*p
        if i != L-1:
            p = p - epsilon*dU(q)
    # half step for momentum    
    p = p - 0.5*epsilon*dU(q)
    # Negate momentum for symmetry
    p = -p
    # Evaluate potential and kinetic energies
    current_U = U(current_q)
    current_K = 0.5*(mp.dot(current_p.T,current_p))
    proposed_U = U(q)
    proposed_K = 0.5*(mp.dot(p.T, p))
    if math.log(random.random()) < (current_U-proposed_U+current_K-proposed_K)[0]:
        return q
    return current_q

def lr_hmc(y, X, epsilon, L, alpha, n_iter):
    def U(beta):
        return mp.sum(mp.log(1 + mp.exp(mp.dot(X, beta))))-mp.dot(y.T,(mp.dot(X,beta)))+(0.5/alpha)*mp.sum(beta**2)

    def dU(beta):
        return mp.dot(X.T, (mp.exp(mp.dot(X,beta))/(1+mp.exp(mp.dot(X,beta))) - y)) + beta/alpha

    D = X.shape[1]
    q = mp.zeros((D, 1), dtype=mp.float32)
    out = mp.zeros((n_iter, D), dtype=mp.float32)
    for i in range(n_iter):
        q = hmc(U, dU, epsilon, L, q)
        out[i,:] = mp.ravel(q)
    return out

with cpu() if args.mode == 'cpu' else gpu(0):
    with open('params.json') as params_file:
        out = {}
        params = json.load(params_file)
        X_train, y_train, X_test, y_test = get_data()
        X_train = mp.array(X_train)
        y_train = mp.array(y_train)
        X_test = mp.array(X_test)
        y_test = mp.array(y_test)
        y_train = mp.expand_dims(y_train, 1)
        z = lr_hmc(y_train, X_train, params['epsilon'], params['n_leaps'], params['alpha'], 1)  # Warm-up
        t = time.perf_counter()
        z = lr_hmc(y_train, X_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'])  
        t = time.perf_counter() - t
        out[f'minpy-{args.mode}'] = t
        coef_ = mp.mean(z[params['burn_in']:], 0)
        acc = mp.mean((sigmoid(mp.dot(X_test, coef_)) > 0.5) == y_test)[0]
        assert acc > 0.8
        print(json.dumps(out))