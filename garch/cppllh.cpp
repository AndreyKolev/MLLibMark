<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++17',  '-Ofast', '-march=native']
%>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numeric>
#include <iterator>
#include <cmath>
namespace py = pybind11;

py::array_t<float> garchSim(py::array_t<float> ret2, py::array_t<float> p) {
  auto h_arr = py::array_t<float>(ret2.size());
  auto h = h_arr.mutable_unchecked<1>();

  auto ret2p = ret2.unchecked<1>();
  auto pp = p.unchecked<1>();

  h[0] = std::accumulate(&ret2p[0], &ret2p[0]+ret2p.shape(0), 0.0)/ret2p.shape(0);

  for (size_t i = 1; i<ret2p.shape(0); i++){
    h[i] = pp[0]+pp[1]*ret2p[i-1]+pp[2]*h[i-1];
  }
  return h_arr;
}

float garchLLH(py::array_t<float> y, py::array_t<float> par){
  auto yp = y.mutable_unchecked<1>();
  auto ret2 = py::array_t<float>(y.size());
  auto ret2p = ret2.mutable_unchecked<1>();

  std::transform(&yp[0], &yp[0]+yp.shape(0), &ret2p[0], [](auto x) {return x * x;});

  auto h = garchSim(ret2, par);
  auto hp = h.mutable_unchecked<1>();

  size_t T = y.size();
  float out = 0;
  for (size_t i = 0; i<T; i++){
    auto tmp = yp[i]/sqrt(hp[i]);
    out += log(hp[i])+tmp*tmp;
  }
  return -0.5*(T-1)*log(2.*M_PI)-0.5*out;
}

PYBIND11_PLUGIN(cppllh){
    pybind11::module m("cppllh", "auto-compiled c++ extension");
    m.def("garchSim", &garchSim);
    m.def("garchLLH", &garchLLH);
    return m.ptr();
}
