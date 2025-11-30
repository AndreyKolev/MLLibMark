"""MC Barrier Option pricing benchmark, Python/mxnet version."""
import math
import json
import argparse
from mxnet import nd, cpu, gpu
from util import benchmark, barrier_data

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
ctx = gpu() if args.mode=='gpu' else cpu() 
out = {}

def pricepaths(s:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> nd.NDArray:
    """
    Simulates log-price paths of a stock under geometric Brownian motion.

    Args:
        : Initial stock price.
        tau: Time to maturity.
        r: Risk-free interest rate.
        q: Dividend yield.
        v: Volatility.
        m: Number of time steps.
        n: Number of simulation paths.

    Returns:
        mxnet.nd.NDArray: Log-price paths of shape (m, n), where each column is a path.
    """
    dt = tau/m
    drift = (r - q - v*v/2)*dt
    scale = v*math.sqrt(dt)
    return math.log(s) + nd.cumsum(nd.random.randn(m, n, loc=drift, scale=scale, dtype='float32'), 0)
    
def barrier(s0:float, k:float, b:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> float:
    """
    Estimates the price of a down-and-out barrier option using Monte Carlo simulation.

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
        float: Estimated option price
    """
    s = pricepaths(s0, tau, r, q, v, m, n)
    l = nd.min(s, 0) > math.log(b)
    payoffs = l*nd.maximum(nd.exp(s[-1]) - k, 0)
    return (math.exp(-r*tau)*nd.mean(payoffs)).asscalar()

data = barrier_data()

with cpu() if args.mode == 'cpu' else gpu(0):
    t = benchmark(lambda: barrier(data['price'], data['strike'],
                  data['barrier'], data['tau'], data['rate'], data['dy'],
                  data['vol'], data['time_steps'], data['n_rep']),
                  data['val'], tol=data['tol'])
    out['mxnet-' + args.mode] = t

print(json.dumps(out))
