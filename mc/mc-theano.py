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
os.environ["THEANO_FLAGS"] = f'mode=FAST_RUN, device={device}, floatX=float32'
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed=0)


def paths(s0, tau, r, q, v, m, n):
    """
    Builds symbolic graph for simulation of log-price paths of a stock under geometric Brownian motion.

    Args:
        s0: Initial stock price.
        tau: Time to maturity.
        r: Risk-free interest rate.
        q: Dividend yield.
        v: Volatility.
        m: Number of time steps.
        n: Number of simulation paths.

    Returns:
        theano.tensor.TensorVariable: Log-price paths of shape (m, n), where each column is a path.
    """
    dt = tau/m  # Time step
    drift = (r - q - v*v/2)*dt  # Drift term
    scale = v*T.sqrt(dt)  # Volatility scaling for Brownian motion
    return T.log(s0) + T.cumsum(srng.normal(avg=drift, std=scale, size=(m, n)), axis=0)


def barrier(s0, k, b, tau, r, q, v, m, n):
    """
    Builds symbolic graph for estimation of the price of a down-and-out barrier option using Monte Carlo simulation.

    Args:
        s0: Initial stock price.
        k: Strike price.
        b: Barrier level.
        tau: Time to maturity (in years).
        r: Risk-free interest rate.
        q: Dividend yield.
        v: Volatility.
        m: Number of time steps.
        n: Number of simulation paths.

    Returns:
        theano.tensor.TensorVariable: Estimated option price
    """
    s = paths(s0, tau, r, q, v, m, n)
    breakpoint()
    #l = T.cast(T.min(s, axis=0) > T.log(b), T.config.floatX)
    l = T.cast(T.any(s > T.log(b), axis=0), T.config.floatX)
    
    # Payoff: max(S_final - K, 0) only if barrier not hit
    payoffs = l*T.maximum(T.exp(s[-1]) - k, 0)
    # Discounted expected payoff
    return T.exp(-r*tau)*T.mean(payoffs)

data = barrier_data()
barrier_t = barrier(data['price'], data['strike'],
                    data['barrier'], data['tau'], data['rate'], data['dy'],
                    data['vol'], data['time_steps'], data['n_rep'])
barrier_fun = function([], barrier_t)

t = benchmark(barrier_fun, data['val'], tol=data['tol'])

out['theano-' + args.mode] = t
print(json.dumps(out))
