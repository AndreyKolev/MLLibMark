"""GARCH LLH benchmark, Theano version."""
import os
import numpy as np
import argparse
import json
from util import benchmark, garch_data

parser = argparse.ArgumentParser()
parser.add_argument('-n', metavar='n', default=1000, type=int,
                    help='number of iterations')
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()

device = 'cuda' if args.mode == 'gpu' else 'cpu'
flags = 'openmp=True, mode=FAST_RUN, device=' + device + ', floatX=float32'
os.environ["THEANO_FLAGS"] = flags

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=0)

out = {}
ret, x0, val_llh = garch_data()

ht0 = np.mean(np.square(ret))
ht = T.scalar('ht')
y = T.vector('y')
lr = T.scalar('lr')
mu = T.scalar('mu')
params = T.vector('params') 

def garch_like(y, ht):
    return params[0] + params[1]*y + params[2]*ht

y2 = T.square(y)
hts, updates = theano.scan(fn=garch_like,
                           sequences=[y2],
                           n_steps=y2.size - 1,
                           outputs_info=ht0)

hts = T.concatenate(([ht0], hts))
tllh = (-0.5 * (y.size - 1) * T.log(2 * np.pi) -
       0.5 * T.sum(T.log(hts) + T.square(y/T.sqrt(hts))))
gllh = T.grad(tllh, wrt=params)
llh = theano.function(inputs=[y, params], outputs=tllh)

t = benchmark(lambda: llh(ret, x0), args.n, val_llh)
out['theano-' + args.mode] = t

def unroll(tx, th0, n):
    th = th0
    th_tmp = []
    for t in range(n):
        th = garch_like(tx[t], th)
        th_tmp.append(th)
    return T.stack(th_tmp)

t_unroll0 = unroll(y2, ht0, len(ret) - 1)
t_unroll = T.concatenate(([ht0], t_unroll0))
t_unroll_out = (-0.5 * (y.size - 1) * T.log(2 * np.pi) -
               0.5 * T.sum(T.log(t_unroll) + T.square(y/T.sqrt(t_unroll))))
tth = theano.function(inputs=[y, params], outputs=t_unroll_out)

t = benchmark(lambda: tth(ret, x0), args.n, val_llh)

out['theano-unroll-' + args.mode] = t
print(json.dumps(out))