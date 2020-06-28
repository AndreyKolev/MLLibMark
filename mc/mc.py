"""MC Barrier Option pricing benchmark, Python version."""

import numpy as np
import math
import json
import os
from numba import jit
import argparse
from multiprocessing import Pool
from functools import partial
from util import benchmark, barrier_data


parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', choices=['std', 'multiprocessing', 'numba', 'c++', 'c++-parallel'],
                    help='use the Numba jit/c++', default='std')
args = parser.parse_args()
out = {}

if args.mode == 'c++':
    os.environ["OMP_NUM_THREADS"] = "1"

def paths(S, tau, r, q, v, M, N):
    """Generate GBM price paths"""
    dt = tau/M
    g1 = (r-q-v/2)*dt
    g2 = np.sqrt(v*dt)
    return S*np.exp(np.cumsum(g1+g2*np.random.randn(M, N).astype(dtype='float32'), 0))
    

def payoff(_, S0, K, B, M, g1, g2):
    """GBM path payout"""
    S = np.exp(math.log(S0)+np.cumsum(g1+g2*np.random.randn(M)))
    l = np.min(S)>B
    #l = np.all(S>B, 0)
    return l*np.max(S[-1]-K, 0)

def barrier(S0, K, B, tau, r, q, v, M, N):
    """Price a barrier option"""
    S = paths(S0, tau, r, q, v, M, N)
    l = np.min(S, 0) > B
    payoffs = l * np.maximum(S[-1, :] - K, 0)
    return np.exp(-r*tau)*np.mean(payoffs)

def barrier_par(S0, K, B, tau, r, q, v, M, N):
    """Price a barrier option (using multiprocessing.Pool)"""
    dt = tau/M
    g1 = (r-q-v/2)*dt
    g2 = np.sqrt(v*dt)
    with Pool() as pool:
        payoffs = pool.map(partial(payoff, S0=S0, K=K, B=B, M=M, g1=g1, g2=g2), range(N))
    return np.exp(-r*tau)*np.mean(payoffs)

data = barrier_data()

if args.mode == 'numba':
    pricepaths = jit(paths)
    barrier = jit(barrier)
elif args.mode == 'multiprocessing':
    barrier = barrier_par
elif args.mode.startswith('c++'):
    import cppimport
    cppcode = cppimport.imp('mc')
    barrier = cppcode.barrier

t = benchmark(lambda: barrier(data['price'], data['strike'],
              data['barrier'], data['tau'], data['rate'], data['dy'],
              data['vol'], data['time_steps'], data['n_rep']),
              data['val'], tol=data['tol'])
out[args.mode if args.mode.startswith('c++') else 'numpy-' + args.mode] = t
print(json.dumps(out))
