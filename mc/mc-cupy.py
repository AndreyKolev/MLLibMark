
import math
import numpy as np
import cupy as cp
import json
import argparse
from util import benchmark, barrier_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='gpu',
                    choices=['std'], help='')
args = parser.parse_args()
out = {}


def pricepaths(S, tau, r, q, v, M, N):
    dt = tau / M
    g1 = (r - q - v / 2) * dt
    g2 = math.sqrt(v * dt)
    aux = math.log(S) + cp.cumsum(g1 + g2 * cp.random.randn(M, N, dtype=cp.float32), 0)
    return cp.exp(aux)


def barrier(S0, K, B, tau, r, q, v, M, N):
    S = pricepaths(S0, tau, r, q, v, M, N)
    l = cp.min(S, 0) > B
    payoffs = l * cp.maximum(S[-1, :] - K, 0)
    return (math.exp(-r * tau) * cp.mean(payoffs)).get().flatten()

data = barrier_data()

t = benchmark(lambda: barrier(data['price'], data['strike'],
              data['barrier'], data['tau'], data['rate'], data['dy'],
              data['vol'], data['time_steps'], data['n_rep'])[0],
              data['val'], tol=data['tol'])
out['cupy'] = t

print(json.dumps(out))