
import math
from time import perf_counter
import json


def benchmark(fun, val_value, tol=1e-1):
    tmp = fun()  # warm-up
    assert math.isclose(tmp, val_value, rel_tol=tol)
    t = perf_counter()
    tmp = fun()
    t = perf_counter() - t
    return t


def barrier_data(path='data.json'):
    with open(path) as data_file:
        data = json.load(data_file)
    return data
