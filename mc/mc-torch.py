"""
Monte Carlo Simulation for Down-and-Out Barrier Option Pricing using PyTorch.
"""
import math
import json
import argparse
import sys
import torch
from util import benchmark, barrier_data


parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', choices=['cpu', 'cuda'], default='cpu')
args = parser.parse_args()

device = torch.device(args.mode)
if args.mode == 'cuda' and not torch.cuda.is_available():
    sys.exit("GPU is not available in this system!")
print(f'Device: {device}')
out = {}


def paths(s0: float, tau: float, r: float, q: float, v: float, m: int, n: int) -> torch.Tensor:
    """Generate GBM price paths"""
    dt = tau/m  # Time step
    drift = (r - q - v**2/2)*dt  # Drift term
    sigma = v*math.sqrt(dt)  # Volatility scaling for Brownian motion
    # Cumulative sum of drift + sigma*dW
    return math.log(s0) + torch.cumsum(drift + sigma*torch.randn(m, n, device=device), dim=0)


def barrier(s0: float, k: float, b: float, tau: float, r: float, q: float, v: float, m: int, n: int) -> float:
    """Price a barrier option"""
    s = paths(s0, tau, r, q, v, m, n)
    not_hit = ~torch.any(s <= math.log(b), 0)
    payoffs = not_hit*torch.relu(s[-1].exp() - k)
    return math.exp(-r*tau)*torch.mean(payoffs)

data = barrier_data()
t = benchmark(lambda: barrier(data['price'], data['strike'],
              data['barrier'], data['tau'], data['rate'], data['dy'],
              data['vol'], data['time_steps'], data['n_rep']),
              data['val'], tol=data['tol'])
out[f'torch-{args.mode}'] = t
print(json.dumps(out))
