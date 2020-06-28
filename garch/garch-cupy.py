"""GARCH LLH benchmark, CUPY version."""
import math
import time
import pandas as pd
import cupy as cp
import json
import logging
import argparse
from util import benchmark, garch_data

parser = argparse.ArgumentParser()
parser.add_argument('-n', metavar='n', default=1000, type=int,
                    help='number of iterations')
parser.add_argument('-mode', metavar='mode', default='gpu',
                    choices=['std'], help='')
args = parser.parse_args()

out = {}

def garchSim(ret2, p):
    h = cp.zeros(ret2.shape[0], dtype='float32')
    h[0] = cp.mean(ret2)
    for i in range(1, ret2.shape[0]):
        h[i] = p[0] + p[1]*ret2[i-1] + p[2]*h[i-1]
    return h


def garchLLH(y, par):
    ret2 = cp.square(y)
    h = garchSim(ret2, par)
    T = y.shape[0]
    llh = -0.5*(T-1)*math.log(2*math.pi) - 0.5*cp.sum(cp.log(h) + (y/cp.sqrt(h))**2)
    return llh


def llh_time():
    ret, x0, val_llh = garch_data()
    ret = cp.array(ret, dtype='float32')
    x0 = cp.array(x0, dtype='float32')

    t = benchmark(lambda: garchLLH(ret, x0), args.n, val_llh)
    print('time:', t)
    return t

out['cupy'] = llh_time()

print(json.dumps(out))