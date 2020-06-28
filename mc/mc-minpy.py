
import minpy.numpy as np
import math
import minpy
import json
import argparse
from minpy.context import cpu, gpu
from util import benchmark, barrier_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
out = {}

def pricepaths(S, tau, r, q, v, M, N):
    dt = tau / M
    g1 = (r - q - v / 2) * dt
    g2 = math.sqrt(v * dt)
    aux = math.log(S) + np.cumsum(g1 + g2 * np.random.randn(M, N, dtype=np.float32), 0)
    return np.exp(aux)


def barrier(S0, K, B, tau, r, q, v, M, N):
    S = pricepaths(S0, tau, r, q, v, M, N)
    l = np.min(S, 0) > B
    payoffs = l * np.maximum(S[-1, :] - K, 0)
    return math.exp(-r*tau) * np.mean(payoffs)

data = barrier_data()

with cpu() if args.mode == 'cpu' else gpu(0):
    t = benchmark(lambda: barrier(data['price'], data['strike'],
                  data['barrier'], data['tau'], data['rate'], data['dy'],
                  data['vol'], data['time_steps'], data['n_rep'])[0],
                  data['val'], tol=data['tol'])
    out['minpy-' + args.mode] = t

print(json.dumps(out))