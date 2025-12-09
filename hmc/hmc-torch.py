"""HMC benchmark, PyTorch v2+ version."""
import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import json
from util import get_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if args.mode == 'gpu' and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Energy function U
@torch.compile
def u(y:torch.Tensor, x:torch.Tensor, beta:torch.Tensor, alpha:float):
    x_beta = x@beta
    return torch.sum(torch.nn.functional.softplus(x_beta)) - y.T@x_beta + (0.5/alpha)*(beta.T@beta)

# Gradient of the energy function U with respect to beta
@torch.compile
def grad_u(y:torch.Tensor, x:torch.Tensor, beta:torch.Tensor, alpha:float):
    return x.T@(torch.sigmoid(x@beta) - y) + beta/alpha


# HMC sampler (single step)
@torch.compile
def hmc_step(
    y: torch.Tensor,
    x: torch.Tensor,
    epsilon: float,
    leapfrog_steps: int,
    current_q: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """Perform a single HMC step using the leapfrog integrator.
    
    Args:
        y: Target values (shape: [n_samples, 1]).
        x: Feature matrix (shape: [n_samples, n_features]).
        epsilon: Step size for the leapfrog integrator.
        leapfrog_steps: Number of leapfrog steps per HMC iteration.
        current_q: Current state (beta vector, shape: [n_features, 1]).
        alpha: Scale of the Gaussian prior.

    Returns:
        New state (beta), either accepted or rejected.
    """
    p = torch.randn_like(current_q, device=current_q.device)
    q = current_q.clone()
    current_p = p.clone()
    # Half-step momentum update
    p = p - 0.5*epsilon*grad_u(y, x, q, alpha)
    # Leapfrog steps
    for i in range(leapfrog_steps):
        # Position update
        q += epsilon*p
        # Momentum update (except last step)
        if i < leapfrog_steps - 1:
            p = p - epsilon*grad_u(y, x, q, alpha)

    # Final momentum flip
    p = 0.5*epsilon*grad_u(y, x, q, alpha) - p

    # Compute acceptance probability
    current_u = u(y, x, current_q, alpha)
    current_k = 0.5*(current_p.T@current_p)
    proposed_u = u(y, x, q, alpha)
    proposed_k = 0.5*(p.T@p)

    log_ratio = current_u - proposed_u + current_k - proposed_k
    log_rand = torch.rand(1, device=current_q.device).log()
    return torch.where(log_rand < log_ratio, q, current_q)

# Main HMC loop
def hmc(
    y: torch.Tensor,
    x: torch.Tensor,
    epsilon: float,
    leapfrog_steps: int,
    alpha: float,
    n_iter: int,
) -> np.ndarray:
    """Run Hamiltonian Monte Carlo sampling for Bayesian logistic regression.

    This function performs HMC sampling to estimate the posterior distribution
    of the regression coefficients.

    Args:
        y: Training target values (shape: [n_samples, 1]).
        x: Training feature matrix (shape: [n_samples, n_features]).
        epsilon: Step size for leapfrog integrator.
        leapfrog_steps: Number of leapfrog steps per HMC iteration.
        alpha: regularization strength.
        n_iter: Total number of HMC iterations to run.

    Returns:
        Numpy array of posterior samples (shape: [n_iter, n_features]).
    """
    samples = []
    current_q = torch.zeros((x.shape[1], 1), device=x.device)

    for _ in range(n_iter):
        current_q = hmc_step(y, x, epsilon, leapfrog_steps, current_q, alpha)
        samples.append(current_q.clone())

    return torch.stack(samples, dim=0).squeeze()


# Load hyper params
with open('params.json') as f:
    params = json.load(f)

x_train, y_train, x_test, y_test = get_data()
y_train = np.expand_dims(y_train, 1)

# Convert to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32, device=device, requires_grad=False)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device, requires_grad=False)

# Warm up
with torch.no_grad():
    hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], 1, burn_in=0)

# Run full HMC
runtime = time.perf_counter()
with torch.no_grad():
    samples = hmc(y_train, x_train, params['epsilon'], params['n_leaps'], params['alpha'], params['n_iter'], burn_in=params['burn_in'])
runtime = time.perf_counter() - runtime

# Estimate posterior mean after burn-in
posterior_mean = np.mean(samples.cpu().numpy()[params['burn_in']:], 0)

# Predict on test set
sigmoid = lambda x: 1.0/(1.0 + np.exp(-x))
acc = np.mean((sigmoid(x_test@posterior_mean) > 0.5) == y_test.squeeze())

# Validate accuracy
assert acc > params['val_accuracy'], f"Accuracy too low: {accuracy}"

# Output result
out = {f'torch-{args.mode}': runtime}
print(json.dumps(out))
