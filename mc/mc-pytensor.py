"""MC Barrier Option pricing benchmark, Python/pytensor version."""

import os
import argparse
import json
import math
from util import benchmark, barrier_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='FAST_RUN',
                    choices=['FAST_RUN', 'NUMBA', 'JAX', 'JAX_CPU'], help='select backend')
args = parser.parse_args()

os.environ['PYTENSOR_FLAGS'] = f'floatX=float32, cast_policy=numpy+floatX, openmp=True, linker=c, optimizer=unsafe'
if args.mode == 'JAX_CPU':
    args.mode = 'JAX'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pytensor
import pytensor.tensor as T
from pytensor.tensor.random.utils import RandomStream
from pytensor import function


out = {}
srng = RandomStream(seed=0)

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
        pytensor.tensor.TensorVariable: Log-price paths of shape (m, n), where each column is a path.
    """
    dt = tau/m  # Time step
    drift = (r - q - v*v/2)*dt  # Drift term
    scale = v*T.sqrt(dt)  # Volatility scaling for Brownian motion
    # Simulate log-price paths
    return T.log(s0) + T.cumsum(srng.normal(loc=drift, scale=scale, size=(m, n)), axis=0)

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
        pytensor.tensor.TensorVariable: Estimated option price
    """
    s = paths(s0, tau, r, q, v, m, n)
    l = T.cast(T.min(s, axis=0) > T.log(b), pytensor.config.floatX)
    # Payoff: max(S_final - K, 0) only if barrier not hit
    payoffs = l*T.maximum(T.exp(s[-1]) - k, 0)
    # Discounted expected payoff
    return T.exp(-r*tau)*T.mean(payoffs)

# Load benchmark data
data = barrier_data()

# Build symbolic graph
barrier_t = barrier(
    s0=data['price'],
    k=data['strike'],
    b=data['barrier'],
    tau=data['tau'],
    r=data['rate'],
    q=data['dy'],
    v=data['vol'],
    m=data['time_steps'],
    n=data['n_rep']
)

# Compile function
barrier_fun = lambda: float(function([], barrier_t, mode=args.mode)())
# Run benchmark
t = benchmark(barrier_fun, data['val'], tol=data['tol'])

out['pytensor-'+args.mode] = t
print(json.dumps(out))
