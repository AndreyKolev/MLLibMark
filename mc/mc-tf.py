"""MC Barrier Option pricing benchmark, Python/TensorFlow version."""

import tensorflow as tf
from util import benchmark, barrier_data
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
out = {}


def paths(S, tau, r, q, v, M, N):
    """Generate GBM price paths"""
    dt = tau / M
    g1 = (r - q - v / 2) * dt
    g2 = tf.sqrt(v * dt)
    return tf.exp(tf.log(S) + tf.cumsum(g1 + g2 * tf.random_normal((M, N))))


def barrier(S0, K, B, tau, r, q, v, M, N):
    """Price a barrier option"""
    S = paths(S0, tau, r, q, v, M, N)
    l = tf.to_float(tf.greater(tf.reduce_min(S, 0), B))
    payoffs = l * tf.maximum(S[-1, :] - K, 0.)
    return tf.exp(-r * tau) * tf.reduce_mean(payoffs)

config = tf.ConfigProto()
if args.mode == 'cpu':
    config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)
    
with tf.Session(config=config) as sess:
    data = barrier_data()
    barr = barrier(data['price'], data['strike'],
                   data['barrier'], data['tau'], data['rate'], data['dy'],
                   data['vol'], data['time_steps'], data['n_rep'])
    t = benchmark(lambda: sess.run(barr), data['val'], tol=data['tol'])
    out['tensorflow-' + args.mode] = t

print(json.dumps(out))
