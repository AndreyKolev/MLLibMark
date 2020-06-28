"""HMC benchmark, CuPy version."""
import numpy as np
import scipy.io
import math
import random
import time
import sklearn.datasets as datasets
from urllib.request import urlretrieve
import tempfile
import json
import argparse
import cupy as cp
from util import get_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='std', choices=['std'], help='')
args = parser.parse_args()
out = {}

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def hmc(U, dU, epsilon, L, current_q):
    q = current_q
    p = cp.random.randn(1, len(current_q), dtype=cp.float32).T
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
    current_K = 0.5*(current_p.T@current_p)
    proposed_U = U(q)
    proposed_K = 0.5*(p.T@p)
    if math.log(random.random()) < (current_U-proposed_U+current_K-proposed_K)[0]:
        return q
    return current_q

def lr_hmc(y, X, epsilon, L, alpha, n_iter):
    def U(beta):
        return cp.sum(cp.log(1 + cp.exp(X@beta))) - y.T@(X@beta) + (0.5/alpha) * cp.sum(beta**2)

    def dU(beta):
        return X.T@(cp.exp(X@beta)/(1+cp.exp(X@beta)) - y) + beta/alpha

    D = X.shape[1]
    q = cp.zeros((D, 1), dtype=cp.float32)
    out = cp.zeros((n_iter, D), dtype=cp.float32)
    for i in range(n_iter):
        q = hmc(U, dU, epsilon, L, q)
        out[i, :] = q.ravel()
    return cp.asnumpy(out)


with open('params.json') as params_file:
    out = {}
    params = json.load(params_file)
    X_train, y_train, X_test, y_test = get_data()
    y_train = cp.array(np.expand_dims(y_train, 1))
    X_train = cp.array(X_train)
    z = lr_hmc(y_train, X_train, params['epsilon'], params['n_leaps'], params['alpha'], 1)  # Warm-up
    t = time.perf_counter()
    z = lr_hmc(y_train, X_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'])  
    t = time.perf_counter() - t
    out[f'cupy'] = t
    coef_ = np.mean(z[params['burn_in']:], 0)
    acc = np.mean((sigmoid(X_test@coef_) > 0.5) == np.squeeze(y_test))
    assert acc > 0.8
    print(json.dumps(out))