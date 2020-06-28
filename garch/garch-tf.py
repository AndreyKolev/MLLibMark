"""GARCH LLH benchmark, Tensorflow version."""

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


def garch_like(ht, y):
    return params[0] + params[1] * y + params[2] * ht

hts = tf.scan(fn=garch_like, elems=y2, initializer=ht_zero)
hts = tf.concat(([ht0], hts[:-1]), 0)


tf_llh = (-0.5*(tf.to_float(tf.shape(y)[0]) - 1) * tf.log(2*np.pi) -
          0.5*tf.reduce_sum(tf.log(hts) + tf.square(y/tf.sqrt(hts))))

with sess.as_default():
    tf.global_variables_initializer().run()
    t = benchmark(lambda: sess.run(tf_llh, feed_dict={params: x0}),
                  args.n,
                  val_llh)
    out['tensorflow-' + args.mode] = t
    
def garch_like(y, ht):
    return params[0] + params[1] * y + params[2] * ht

def unroll(tx, th0, n):
    th = th0
    th_tmp = []
    for t in range(n):
        th = garch_like(tx[t], th)
        th_tmp.append(th)
    return tf.stack(th_tmp)

tf_unroll0 = unroll(y2, ht0, len(ret) - 1)
hts = tf.concat(([ht0], tf_unroll0), 0)
tf_llh_unroll = (-0.5*(tf.to_float(tf.shape(y)[0])-1) * tf.log(2*np.pi) -
                 0.5*tf.reduce_sum(tf.log(hts) + tf.square(y/tf.sqrt(hts))))


with sess.as_default():
    tf.global_variables_initializer().run()
    t = benchmark(lambda: sess.run(tf_llh_unroll, feed_dict={params: x0}),
                  args.n,
                  val_llh)
    out['tensorflow-' + args.mode + '-unroll'] = t
    
print(json.dumps(out))