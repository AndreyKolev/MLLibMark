<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++17', '-Ofast', '-fopenmp', '-march=native']
cfg['linker_args'] = ['-lgomp']
%>
#define _USE_MATH_DEFINES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numeric>
#include <iterator>
#include <cmath>
#include <random>
#include <cstdlib>
#include <omp.h>
#include <limits> 

namespace py = pybind11;

// Julia is using DSFMT rnd
py::array_t<float> pricepaths(float S, float tau, float r, float q, float v, size_t M, size_t N) {
	auto dt = tau/M;
	auto g1 = (r-q-v/2)*dt;
	auto g2 = sqrt(v*dt);

	auto gbm_array = py::array_t<float>({M, N});
	auto gbm = gbm_array.mutable_unchecked<2>();

	std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);
    for (size_t j = 0; j < gbm_array.shape(1); j++){
        float cumsum = log(S);
        for (size_t i = 0; i < gbm_array.shape(0); i++){
        	cumsum += g1 + g2*dist(gen);
        	gbm(i, j) = exp(cumsum);
        }
    }
    return gbm_array;
}

float barrier(float S, float K, float B, float tau, float r, float q, float v, size_t M, size_t N) {
	auto dt = tau/M;
	auto g1 = (r-q-v/2)*dt;
	auto g2 = sqrt(v*dt);

    float payoffs = 0;

    #pragma omp parallel for reduction(+:payoffs)
    for (size_t i=0; i<N; i++){
    	std::random_device rd;
    	std::mt19937 gen(rd());
    	std::normal_distribution<float> dist(0, 1);
    
        float cumsum = log(S);
        float min_price = std::numeric_limits<float>::max();
        for (size_t j=0; j<M; j++){
        	cumsum += g1 + g2*dist(gen);
        	float price = exp(cumsum);
        	if(price < min_price){
        		min_price = price;
        	}       	
        }
        if (min_price > B)
        	payoffs += std::max(exp(cumsum)-K, 0.f);
    }
    return exp(-r*tau)*payoffs/N;
}


PYBIND11_PLUGIN(mc) {
    pybind11::module m("mc", "auto-compiled c++ extension");
    m.def("barrier", &barrier);
    return m.ptr();
}
