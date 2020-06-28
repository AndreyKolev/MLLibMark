"""GARCH LLH benchmark utility functions"""
import math
from time import perf_counter
import numpy as np
import json


def benchmark(fun, n, val_value, tol=1e-3):
    tmp = fun()  # warm-up
    assert math.isclose(tmp, val_value, rel_tol=tol)
    t = perf_counter()
    for _ in range(n):
        tmp = fun()
    t = perf_counter() - t
    return t


def garch_data(path='data.json'):
    with open(path) as data_file:
        data = json.load(data_file)
    #ret = np.diff(np.log(np.array(data['price'], dtype='float32')))
    #ret = ret - np.mean(ret)
    ret = np.array(data['ret'], dtype='float32')
    x0 = np.array(data['x0'], dtype='float32')
    return ret, x0, data['llh']
