"""MC Barrier Option pricing benchmark, Python/TensorFlow v2 version."""

import tensorflow as tf
from util import benchmark, barrier_data
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
if args.mode == 'cpu':
    tf.config.set_visible_devices([], 'GPU')  # Use CPU

out = {}

def paths(S, tau, r, q, v, M, N):
    """Generate GBM price paths"""
    dt = tau/M
    g1 = (r-q-v/2)*dt
    g2 = tf.sqrt(v*dt)
    return tf.exp(tf.math.log(S) + tf.cumsum(g1+g2*tf.random.normal((M, N))))

@tf.function
def barrier(S0, K, B, tau, r, q, v, M, N):
    """Price a barrier option"""
    S = paths(S0, tau, r, q, v, M, N)
    l = tf.cast(tf.greater(tf.reduce_min(S, 0), B), dtype=tf.float32)
    payoffs = l*tf.maximum(S[-1, :]-K, 0)
    return tf.exp(-r*tau)*tf.reduce_mean(payoffs)

data = barrier_data()

def barrier_fun():
    return barrier(data['price'], data['strike'],
        data['barrier'], data['tau'], data['rate'], data['dy'],
        data['vol'], data['time_steps'], data['n_rep'])

t = benchmark(barrier_fun, data['val'], tol=data['tol'])
out['tensorflow-'+args.mode] = t

print(json.dumps(out))
