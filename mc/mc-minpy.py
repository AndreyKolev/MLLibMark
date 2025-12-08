"""MC Barrier Option pricing benchmark, Python/minpy version."""
import minpy.numpy as mp
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

def pricepaths(s0:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> minpy.numpy.ndarray:
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
        minpy.numpy.ndarray: Log-price paths of shape (m, n), where each column is a path.
    """
    dt = tau/m
    drift = (r - q - v*v/2)*dt  # Drift term
    scale = v*math.sqrt(dt)  # Volatility scaling for Brownian motion
    aux = math.log(S) + mp.cumsum(drift + scale*mp.random.randn(M, N, dtype=mp.float32), 0)
    return mp.exp(aux)


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
    l = mp.min(s, 0) > b
    payoffs = l * mp.maximum(s[-1] - k, 0)
    return math.exp(-r*tau)*mp.mean(payoffs)

data = barrier_data()

with cpu() if args.mode == 'cpu' else gpu(0):
    t = benchmark(lambda: barrier(data['price'], data['strike'],
                  data['barrier'], data['tau'], data['rate'], data['dy'],
                  data['vol'], data['time_steps'], data['n_rep'])[0],
                  data['val'], tol=data['tol'])
    out['minpy-' + args.mode] = t

print(json.dumps(out))
