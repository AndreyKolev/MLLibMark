"""GARCH LLH benchmark, Python version."""
import os

import math
import time
import numpy as np
import json
import argparse
from numba import jit
from util import benchmark, garch_data

parser = argparse.ArgumentParser()
parser.add_argument('-n', metavar='n', default=1000, type=int,
                    help='number of iterations')
parser.add_argument('-mode', metavar='mode', default='std',
                    choices=['std', 'numba', 'c++'], help='use the Numba jit')
args = parser.parse_args()


def garchSim(ret2, p):
    h = np.zeros_like(ret2)
    h[0] = np.mean(ret2)
    for i in range(1, len(ret)):
        h[i] = p[0] + p[1]*ret2[i-1] + p[2]*h[i-1]
    return h


def garchLLH(y, par):
    h = garchSim(np.square(y), par)
    t = len(y)
    return -0.5*(t-1)*np.log(2*math.pi)-0.5*np.sum(np.log(h)+(y/np.sqrt(h))**2)


if args.mode == 'numba':
    garchSim = jit(garchSim)
    garchLLH = jit(garchLLH)

out = {}
ret, x0, val_llh = garch_data()


if args.mode == 'c++':
    import cppimport
    cpp_llh = cppimport.imp('cppllh')
    out['c++'] = benchmark(lambda: cpp_llh.garchLLH(ret, x0), args.n, val_llh)
    print('c++ time:', out['c++'])
else:
    out['numpy-' + args.mode] = benchmark(lambda: garchLLH(ret, x0), args.n,
                                          val_llh)
print(json.dumps(out))
