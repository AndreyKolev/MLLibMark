"""MC Barrier Option pricing benchmark, Python/TensorFlow v1 (compat.v1) version."""

from util import benchmark, barrier_data
import argparse
import json
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
out = {}

def paths(S, tau, r, q, v, m, n):
    """Generate GBM price paths"""
    dt = tau/m
    drift = (r - q - v*v/2)*dt
    scale = v*tf.sqrt(dt)
    return tf.log(S) + tf.cumsum(tf.random_normal((m, n), mean=drift, stddev=scale))

def barrier(s0, k, b, tau, r, q, v, m, n):
    """Price a barrier option"""
    s = paths(s0, tau, r, q, v, m, n)
    payoffs = tf.where(tf.reduce_min(s, 0) <= tf.log(b), tf.zeros(n), tf.nn.relu(tf.exp(s[-1]) - k))
    return tf.exp(-r*tau)*tf.reduce_mean(payoffs)

config = tf.ConfigProto()
if args.mode == 'cpu':
    config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)
    
with tf.Session(config=config) as sess:
    data = barrier_data()
    barr = barrier(data['price'], data['strike'],
                   data['barrier'], data['tau'], data['rate'], data['dy'],
                   data['vol'], data['time_steps'], data['n_rep'])
    t = benchmark(lambda: sess.run(barr), data['val'], tol=data['tol'])
    out['tensorflow-v1-'+args.mode] = t

print(json.dumps(out))
