"""HMC benchmark, numpy version."""
import numpy as np
import math
import random
import time
import json
import argparse
from util import get_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='std', choices=['std'], help='')
args = parser.parse_args()
out = {}

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def hmc(U, dU, epsilon, L, current_q):
    q = current_q
    p = np.random.randn(1, len(current_q)).astype('float32').T
    current_p = p
    # half step for momentum
    p = p - 0.5*epsilon*dU(q)
    # pos and momentum steps
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
        return np.sum(np.log(1 + np.exp(X@beta))) - y.T@(X@beta) + (0.5/alpha) * np.sum(beta**2)

    def dU(beta):
        return X.T@(np.exp(X@beta)/(1+np.exp(X@beta)) - y) + beta/alpha

    D = X.shape[1]
    q = np.zeros((D, 1), dtype='float32')
    out = np.zeros((n_iter, D), dtype='float32')
    for i in range(n_iter):
        q = hmc(U, dU, epsilon, L, q)
        out[i, :] = q.ravel()
    return out


with open('params.json') as params_file:
    out = {}
    params = json.load(params_file)
    X_train, y_train, X_test, y_test = get_data()
    y_train = np.expand_dims(y_train, 1)
    z = lr_hmc(y_train, X_train, params['epsilon'], params['n_leaps'], params['alpha'], 1)  # Warm-up
    t = time.perf_counter()
    z = lr_hmc(y_train, X_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'])  
    t = time.perf_counter()-t
    out[f'numpy-std'] = t
    coef_ = np.mean(z[params['burn_in']:], 0)
    acc = np.mean((sigmoid(X_test@coef_) > 0.5) == np.squeeze(y_test))
    assert acc > 0.8
    print(json.dumps(out))