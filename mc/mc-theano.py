"""MC Barrier Option pricing benchmark, Python/Theano version."""

import os
import argparse
import json
from util import benchmark, barrier_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
out = {}

device = 'cuda' if args.mode == 'gpu' else 'cpu'
flags = 'openmp=True, mode=FAST_RUN, device=' + device + ', floatX=float32'
os.environ["THEANO_FLAGS"] = flags

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed=0)


def paths(S, tau, r, q, v, M, N):
    """Generate GBM price paths"""
    dt = tau / M
    g1 = (r - q - v / 2) * dt
    g2 = T.sqrt(v * dt)
    return T.exp(T.log(S) + T.cumsum(g1 + g2 * srng.normal((M, N)), 0))


def barrier(S0, K, B, tau, r, q, v, M, N):
    """Price a barrier option"""
    S = paths(S0, tau, r, q, v, M, N)
    l = T.cast(T.min(S, 0) > B, T.config.floatX)  # T.switch?
    payoffs = l * T.maximum(S[-1, :] - K, 0)
    return T.exp(-r * tau) * T.mean(payoffs)

data = barrier_data()
barrier_t = barrier(data['price'], data['strike'],
                    data['barrier'], data['tau'], data['rate'], data['dy'],
                    data['vol'], data['time_steps'], data['n_rep'])
barrier_fun = function([], barrier_t)

t = benchmark(barrier_fun, data['val'], tol=data['tol'])

out['theano-' + args.mode] = t
print(json.dumps(out))