"""
MC Barrier Option pricing benchmark, Python + C++/Pybind11 via cppimport version.

This script benchmarks the Monte Carlo pricing of a barrier option using multiple execution modes:
- Standard NumPy
- NumPy with Multiprocessing (via multiprocessing.Pool)
- C++ implementations using Pybind11 and cppimport:
  - Single-threaded C++
  - OpenMP-parallel C++
  - C++ parallel using stdlib concurrency support library

The C++ code is compiled at runtime using cppimport
"""

import math
import json
import os
import argparse
from multiprocessing import Pool
from functools import partial
import numpy as np
from util import benchmark, barrier_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', choices=['std', 'multiprocessing', 'c++', 'c++-openmp', 'c++-threads'], help='execution mode', default='std')
args = parser.parse_args()
out = {}


def paths(s:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> np.ndarray:
    """
    Generate geometric Brownian motion (GBM) price paths using Monte Carlo simulation.

    This function simulates log-price paths of a financial asset under the GBM model,
    where the logarithm of the asset price follows a drifted Brownian motion.

    Args:
        s (float): Initial asset price (S₀).
        tau (float): Time to maturity (in years).
        r (float): Risk-free interest rate (annualized).
        q (float): Dividend yield (annualized).
        v (float): Volatility (annualized).
        m (int): Number of time steps per path.
        n (int): Number of simulated paths.

    Returns:
        np.ndarray: Log-price paths of shape (m, n), where each column represents
                    a simulated path of log(S_t).
    """
    dt = tau/m
    drift = (r - q - v*v/2)*dt
    scale = v*math.sqrt(dt)
    rng = np.random.default_rng()
    return math.log(s) + np.cumsum(drift + scale*rng.standard_normal((m, n), np.float32), 0)

def barrier(s0:float, k:float, b:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> float:
    """
    Price a down-and-out barrier option using Monte Carlo simulation.

    This function computes the expected discounted payoff of a European-style down-and-out
    barrier option via Monte Carlo methods.

    Args:
        s0 (float): Initial asset price (S₀).
        k (float): Strike price of the option.
        b (float): Barrier level (must be below current price for down-and-out).
        tau (float): Time to maturity (in years).
        r (float): Risk-free interest rate (annualized).
        q (float): Dividend yield (annualized).
        v (float): Volatility (annualized).
        m (int): Number of time steps per path.
        n (int): Number of simulated paths.

    Returns:
        float: Estimated option price
    """
    s = paths(s0, tau, r, q, v, m, n)
    l = np.min(s, 0) > math.log(b)
    payoffs = l*np.maximum(np.exp(s[-1,:]) - k, 0)
    return np.exp(-r*tau)*np.mean(payoffs)

def payoff(_, s0:float, k:float, b:float, m:int, drift:float, scale:float) -> float:
    """
    Compute the payoff of a single GBM path for a down-and-out barrier option.

    This function evaluates the payoff for one simulated path, checking whether the
    barrier condition is violated. It is designed for use with multiprocessing.

    Args:
        _ (any): Placeholder argument to match signature expected by `Pool.map`.
        s0 (float): Initial asset price (S₀).
        k (float): Strike price.
        b (float): Barrier level.
        m (int): Number of time steps per path.
        drift (float): Drift term per time step: (r - q - v²/2) * dt.
        scale (float): Scale factor for volatility: v * sqrt(dt).

    Returns:
        float: Payoff value (0 if barrier hit, otherwise max(S_T - K, 0)).
    """
    rng = np.random.default_rng()
    s = math.log(s0) + np.cumsum(drift + scale*rng.standard_normal((m), np.float32))
    l = np.min(s, 0) > math.log(b)
    #l = np.all(s > math.log(b), 0)
    #l = ~np.any(s < math.log(b), 0)
    return l*np.max(np.exp(s[-1]) - k, 0)
    
def barrier_par(s0:float, k:float, b:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> float:
    """
    Price a down-and-out barrier option using multiprocessing for parallel Monte Carlo simulation.

    This function leverages `multiprocessing.Pool` to compute payoffs across multiple
    independent paths in parallel, improving performance on multi-core systems.

    Args:
        s0 (float): Initial asset price (S₀).
        k (float): Strike price.
        b (float): Barrier level.
        tau (float): Time to maturity (in years).
        r (float): Risk-free interest rate (annualized).
        q (float): Dividend yield (annualized).
        v (float): Volatility (annualized).
        m (int): Number of time steps per path.
        n (int): Number of simulated paths.

    Returns:
        float: Estimated option price.
    """
    dt = tau/m
    drift = (r - q - v*v/2)*dt
    scale = v*math.sqrt(dt)
    with Pool() as pool:
        payoffs = pool.map(partial(payoff, s0=s0, k=k, b=b, m=m, drift=drift, scale=scale), range(n))
    return np.exp(-r*tau)*np.mean(payoffs)

data = barrier_data()

if args.mode == 'multiprocessing':
    barrier = barrier_par
elif args.mode.startswith('c++'):
    import cppimport
    cppcode = cppimport.imp('mc')
    if args.mode == 'c++':
        barrier = cppcode.barrier
    if args.mode == 'c++-openmp':
        barrier = cppcode.barrier_omp
    elif args.mode == 'c++-threads':
        barrier = cppcode.barrier_threads

t = benchmark(lambda: barrier(data['price'], data['strike'],
              data['barrier'], data['tau'], data['rate'], data['dy'],
              data['vol'], data['time_steps'], data['n_rep']),
              data['val'], tol=data['tol'])
out[args.mode if args.mode.startswith('c++') else 'numpy-' + args.mode] = t
print(json.dumps(out))
