import math
from time import perf_counter
import json
from typing import Callable

"""
Benchmarking utility function to measure the execution time of a given function.

This function runs the provided function twice: first to warm up (discard the result), 
then measures the actual execution time of a second run. It also verifies that the 
function's output is close to the expected value within a specified relative tolerance.

Args:
    fun (callable): The function to benchmark. Must return a numeric value.
    val_value (float): The expected value to compare against.
    tol (float, optional): Relative tolerance for comparison. Defaults to 1e-1 (10%).

Returns:
    float: Execution time in seconds of the second run of `fun`.

Raises:
    AssertionError: If the function's output is not within the specified tolerance of `val_value`.
"""
def benchmark(fun:Callable, val_value:float, tol:float=1e-1):
    tmp = fun()  # warm-up
    assert math.isclose(tmp, val_value, rel_tol=tol)
    t = perf_counter()
    tmp = fun()
    t = perf_counter() - t
    return t


"""
Loads option simulation parameters from the JSON file.

Args:
    path (str, optional): Path to the JSON file. Defaults to 'data.json'.

Returns:
    dict: Parsed JSON data from the file.
"""
def barrier_data(path:str='data.json') -> dict:
    with open(path) as data_file:
        data = json.load(data_file)
    return data
