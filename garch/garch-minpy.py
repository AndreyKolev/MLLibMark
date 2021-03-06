"""GARCH LLH benchmark, MINPY version."""
import math
import time
import minpy
import minpy.numpy as np
import pandas as pd
from minpy.context import set_context, cpu, gpu
import json
import logging
import argparse
from util import benchmark, garch_data

parser = argparse.ArgumentParser()
parser.add_argument('-n', metavar='n', default=1000, type=int,
                    help='number of iterations')
parser.add_argument('-mode', metavar='mode', default='cpu',
                    choices=['cpu', 'gpu'], help='use cpu/gpu')
args = parser.parse_args()

out = {}

def garchSim(ret2, p):
    h = np.zeros(ret2.shape[0], dtype='float32')
    h[0] = np.mean(ret2)
    for i in range(1, ret2.shape[0]):
        h[i] = p[0] + p[1]*ret2[i-1] + p[2]*h[i-1]
    return h


def garchLLH(y, par):
    h = garchSim(np.square(y), par)
    T = y.shape[0]
    llh = -0.5*(T-1)*math.log(2*math.pi) - 0.5*np.sum(np.log(h) + (y/np.sqrt(h))**2)
    return llh[0]


def llh_time():
    ret, x0, val_llh = garch_data()
    ret = np.array(ret)
    x0 = np.array(x0)

    t = benchmark(lambda: garchLLH(ret, x0), args.n, val_llh)
    return t

with cpu() if args.mode == 'cpu' else gpu(0):
    out['minpy-' + args.mode] = llh_time()

print(json.dumps(out))