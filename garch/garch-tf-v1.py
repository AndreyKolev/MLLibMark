"""GARCH LLH benchmark, Tensorflow v1 version."""

import numpy as np
import time
import argparse
import json
import math
from util import benchmark, garch_data
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('-n', metavar='n', default=1000, type=int,
                    help=f'number of iterations')
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()

out = {}
ret, x0, val_llh = garch_data()

config = tf.ConfigProto()
if args.mode == 'cpu':
    config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)

sess = tf.Session(config=config)

ht0 = np.mean(np.square(ret))
y = tf.Variable(ret, name='y', dtype='float32')
params = tf.Variable(x0, name='params')
y2 = tf.square(y)
ht_zero = tf.reduce_mean(y2)

def garch(ht, y):
    return params[0] + params[1]*y + params[2]*ht

hts = tf.scan(fn=garch, elems=y2, initializer=ht_zero)
hts = tf.concat(([ht0], hts[:-1]), 0)
tf_llh = (-0.5*(tf.cast(y.shape[0], tf.float32)-1)*tf.log(2*np.pi) -
          0.5*tf.reduce_sum(tf.log(hts) + tf.square(y/tf.sqrt(hts))))

with sess.as_default():
    tf.global_variables_initializer().run()
    t = benchmark(lambda: sess.run(tf_llh, feed_dict={params: x0}),
                  args.n,
                  val_llh)
    out['tensorflow-' + args.mode] = t
    
def unroll(x, h0, n):
    h = h0
    return tf.stack([h := garch(h, x[t]) for t in range(n)])
    
tf_unroll0 = unroll(y2, ht0, len(ret) - 1)
hts = tf.concat(([ht0], tf_unroll0), 0)
tf_llh_unroll = (-0.5*(tf.cast(y.shape[0], tf.float32)-1)*tf.log(2*np.pi) -
                 0.5*tf.reduce_sum(tf.log(hts) + tf.square(y/tf.sqrt(hts))))

with sess.as_default():
    tf.global_variables_initializer().run()
    t = benchmark(lambda: sess.run(tf_llh_unroll, feed_dict={params: x0}),
                  args.n,
                  val_llh)
    out['tensorflow-v1-'+args.mode+'-unroll'] = t
    
print(json.dumps(out))