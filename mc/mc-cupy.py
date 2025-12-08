"""MC Barrier Option pricing benchmark, Python/CuPy version."""
import math
import numpy as np
import cupy as cp
import json
import argparse
from util import benchmark, barrier_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='gpu',
                    choices=['gpu'], help='')
args = parser.parse_args()
out = {}


def pricepaths(s0:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> cp.ndarray:
    """
    Simulates log-price paths of a stock under geometric Brownian motion.

    Args:
        s0: Initial stock price.
        tau: Time to maturity.
        r: Risk-free interest rate.
        q: Dividend yield.
        v: Volatility.
        m: Number of time steps.
        n: Number of simulation paths.

    Returns:
        cupy.ndarray: Log-price paths of shape (m, n), where each column is a path.
    """
    dt = tau/M
    drift = (r - q - v*v/2)*dt
    scale = v*math.sqrt(dt)
    return math.log(s0) + cp.cumsum(drift + scale*cp.random.randn(m, n, dtype=cp.float32), 0)

def barrier(s0:float, k:float, b:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> float:
    """
    Estimates the price of a down-and-out barrier option using Monte Carlo simulation.

    The option becomes worthless if the stock price hits the barrier level `b` during
    the life of the option.

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
        float: Estimated option price.
    """
    s = pricepaths(s0, tau, r, q, v, m, n)
    payoffs = cp.where(cp.any(s <= cp.log(b), 0), .0, cp.relu(cp.exp(s[-1]) - k))
    return (math.exp(-r*tau)*cp.mean(payoffs)).get().flatten()

data = barrier_data()

t = benchmark(lambda: barrier(data['price'], data['strike'],
              data['barrier'], data['tau'], data['rate'], data['dy'],
              data['vol'], data['time_steps'], data['n_rep'])[0],
              data['val'], tol=data['tol'])
out['cupy'] = t

print(json.dumps(out))
