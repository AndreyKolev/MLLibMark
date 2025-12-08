"""
Monte Carlo Simulation for Down-and-Out Barrier Option Pricing using JAX.
"""
import json
import argparse
import jax.numpy as jnp
import jax
from util import benchmark, barrier_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', choices=['std'], default='std')
args = parser.parse_args()
out = {}


def pricepaths(s: float, tau: float, r: float, q: float, v: float, m: int, n: int) -> jnp.array:
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
        jnp.ndarray: Log-price paths of shape (m, n), where each column is a path.
    """
    key = jax.random.key(0)
    dt = tau/m  # Time step
    drift = (r - q - v*v/2)*dt  # Drift term
    scale = v*jnp.sqrt(dt)  # Volatility scaling for Brownian motion
    # Simulate log-price paths
    return jnp.log(s) + jnp.cumsum(drift + scale*jax.random.normal(key, (m, n)), 0)


def barrier(s0: float, k: float, b: float, tau: float, r: float, q: float, v: float, m: int, n: int) -> float:
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
    s = pricepaths(s0, tau, r, q, v, m, n)  # Generate simulated stock price paths
    payoffs = jnp.where(jnp.min(s, 0) <= jnp.log(b), 0, jax.nn.relu(jnp.exp(s[-1]) - k))
    return (jnp.exp(-r*tau)*jnp.mean(payoffs))  # Discount expected payoff


if __name__ == "__main__":
    # JIT-compile
    pricepaths = jax.jit(pricepaths, static_argnames=['m', 'n'])
    barrier = jax.jit(barrier, static_argnames=['m', 'n'])

    # Load benchmark data
    data = barrier_data()

    # Run benchmark
    t = benchmark(lambda: barrier(data['price'], data['strike'],
                                  data['barrier'], data['tau'], data['rate'], data['dy'],
                                  data['vol'], data['time_steps'], data['n_rep']).item(),
                                  data['val'], tol=data['tol'])
    out[f'jax-{args.mode}'] = t
    print(json.dumps(out))
