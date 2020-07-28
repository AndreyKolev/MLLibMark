"""GARCH LLH benchmark, Tensorflow v2 version."""

import numpy as np
import time
import tensorflow as tf
import argparse
import json
import math
from util import benchmark, garch_data

parser = argparse.ArgumentParser()
parser.add_argument('-n', metavar='n', default=1000, type=int,
                    help=f'number of iterations')
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()

if args.mode == 'cpu':
    tf.config.set_visible_devices([], 'GPU')  # Use CPU

out = {}

ret, x0, val_llh = garch_data()

@tf.function
def llh(y, params):
    def garch(ht, y):
        return params[0] + params[1]*y + params[2]*ht    
    
    y2 = tf.square(y)
    ht0 = tf.reduce_mean(y2)
    hts = tf.scan(fn=garch, elems=y2, initializer=ht0)
    hts = tf.concat(([ht0], hts[:-1]), 0)
    return (-0.5*(tf.cast(y.shape[0], tf.float32)-1)*tf.math.log(2*np.pi) - 
        0.5*tf.reduce_sum(tf.math.log(hts) + tf.square(y/tf.sqrt(hts))))

@tf.function    
def llh_unroll(y, params):
    def garch(ht, y):
        return params[0] + params[1]*y + params[2]*ht

    y2 = tf.square(y)
    ht0 = tf.reduce_mean(y2)
    h = ht0
    hts = []
    for t in range(len(y2)-1):
        h = garch(h, y2[t])
        hts.append(h)
    # hts0 = [h := garch_like(h, y2[t]) for t in range(len(y2)-1)]
    hts = tf.concat(([ht0], tf.stack(hts)), 0)
    return (-0.5*(tf.cast(y.shape[0], tf.float32)-1)*tf.math.log(2*np.pi) - 
            0.5*tf.reduce_sum(tf.math.log(hts) + tf.square(y/tf.sqrt(hts))))

t = benchmark(lambda: llh(ret, x0).numpy(), args.n, val_llh)
out['tensorflow-'+args.mode] = t

t = benchmark(lambda: llh_unroll(ret, x0).numpy(), args.n, val_llh)
out['tensorflow-' + args.mode + '-unroll'] = t

print(json.dumps(out))