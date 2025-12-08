# Computational & Deep Learning Frameworks Benchmark on real-world tasks

### Use *benchmark.ipynb* to run the benchmark.

Frameworks compared:
* [Python 3](https://www.python.org/)/[Numpy](http://www.numpy.org/) with or without [Numba JIT compilation](http://numba.pydata.org/)
* [Julia](https://julialang.org/) The Julia Programming Language
* C++ called from Python using [pybind11](https://github.com/pybind/pybind11)
* [Rust](https://rust-lang.org/), also using [Rayon: A data parallelism library for Rust](https://github.com/rayon-rs/rayon) and [ndarray](https://github.com/rust-ndarray/ndarray) crates
* [R](https://www.r-project.org/) Project for Statistical Computing
* [PyTorch](https://pytorch.org/)
* [Tensorflow](https://www.tensorflow.org) v1 and v2 in CPU/GPU mode
* [JAX](https://docs.jax.dev/en/latest/)
* [Theano](https://github.com/Theano/Theano)
* [PyTensor](https://github.com/pymc-devs/pytensor)
* [Apache MXNet/NDArray](https://mxnet.apache.org) NDArray library in Apache MXNet.
* [MinPy](https://github.com/dmlc/minpy) a NumPy interface above [Apache MXNet](https://mxnet.apache.org) backend.
* [CuPy](https://cupy.chainer.org/) an open-source matrix library accelerated with NVIDIA CUDA.


# Tasks:
*See benchmark.ipynb notebook for details.*

*Benchmark output on hexacore CPU / Kepler GPU*

## Task I: Time-series model (GARCH / *Scan* operation  benchmark)
![Task I](img/task1.png)

## Task II: Monte Carlo simulation (Barrier option pricing)
![Task II](img/task2.png)

## Task III: Logistic regression using Hybrid Monte Carlo (HMC)
![Task III](img/task3.png)
