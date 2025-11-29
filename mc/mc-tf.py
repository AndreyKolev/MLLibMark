"""MC Barrier Option pricing benchmark, Python/TensorFlow v2 version."""

import tensorflow as tf
from util import benchmark, barrier_data
import argparse
import json
import sys
import math

parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()
if args.mode == 'cpu':
    tf.config.set_visible_devices([], 'GPU')
elif args.mode == 'gpu':
	gpus = tf.config.list_physical_devices('GPU')
	if not gpus:
		sys.exit("GPU is not available in this system!")
out = {}

def paths(s:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> tf.Tensor:
    """Generate GBM price paths"""
    dt = tau/m
    drift = (r - q - v*v/2)*dt
    scale = v*tf.sqrt(dt)
    return tf.math.log(s) + tf.cumsum(tf.random.normal((m, n), mean=drift, stddev=scale))

@tf.function
def barrier(s0:float, k:float, b:float, tau:float, r:float, q:float, v:float, m:int, n:int) -> float:
    """Price a barrier option"""
    s = paths(s0, tau, r, q, v, m, n)
    payoffs = tf.where(tf.reduce_min(s, 0) <= math.log(b), .0, tf.nn.relu(tf.exp(s[-1]) - k))    
    return math.exp(-r*tau)*tf.reduce_mean(payoffs)

data = barrier_data()
def barrier_fun():
    return barrier(data['price'], data['strike'],
        data['barrier'], data['tau'], data['rate'], data['dy'],
        data['vol'], data['time_steps'], data['n_rep'])

t = benchmark(barrier_fun, data['val'], tol=data['tol'])
out['tensorflow-' + args.mode] = t

print(json.dumps(out))
