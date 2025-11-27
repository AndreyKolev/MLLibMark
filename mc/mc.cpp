// =============================================================================
// Monte Carlo Simulation for Barrier Options (C++ with pybind11)
// =============================================================================
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++23', '-O3', '-fopenmp', '-march=native']
cfg['linker_args'] = ['-lgomp']
%>
// CLANG OMP LINKER FLAG
//cfg['linker_args'] = ['-lomp'] 

#include <numeric>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <random>
#include <cstdlib>
#include <limits>
#include <atomic>
#include <thread>
#include <future>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * @brief Single-thread Monte Carlo simulation for a down-and-out barrier option using geometric Brownian motion.
 *
 * This function simulates N paths of a stock price under geometric Brownian motion,
 * checking whether the price hits a lower barrier `b` during its path. If the barrier
 * is hit, the option is knocked out (payoff = 0). Otherwise, the payoff is max(S_T - K, 0).
 *
 * @param s      Initial stock price
 * @param k      Strike price
 * @param b      Barrier level (down-and-out)
 * @param tau    Time to maturity (in years)
 * @param r      Risk-free interest rate
 * @param q      Dividend yield
 * @param v      Volatility (annualized)
 * @param m      Number of time steps in each path
 * @param n      Number of Monte Carlo paths to simulate
 *
 * @return       Expected discounted payoff (option price)
 */

float barrier(float s, float k, float b, float tau, float r, float q, float v, size_t m, size_t n) {
    const float dt = tau/m;
    const float drift = (r - q - v*v/2)*dt;
    const float scale = v*std::sqrt(dt);
    const float logb = std::log(b);
    const float log_s0 = std::log(s);
    float payoffs = 0.0;

	std::random_device rd;
	std::mt19937 gen(rd());
    std::normal_distribution<float> dist(drift, scale);
        
    for (size_t i = 0; i < n; i++) {
        float log_price = log_s0;
        bool barrier_hit = false;
        for (size_t j = 0; j < m; j++) {
            log_price += dist(gen);
            if (log_price < logb) {
                barrier_hit = true;
                break;
            }
        }
        if (!barrier_hit) { // Accumulate payoff if barrier not hit
        	payoffs += std::max(std::exp(log_price) - k, 0.f);
        }
    }
    // Calculate discounted expected payoff (option price)
    return std::exp(-r*tau)*payoffs/n;
}

/**
 * @brief Monte Carlo simulation (OpenMP version) for a down-and-out barrier option using geometric Brownian motion.
 *
 * This function simulates N paths of a stock price under geometric Brownian motion,
 * checking whether the price hits a lower barrier `b` during its path. If the barrier
 * is hit, the option is knocked out (payoff = 0). Otherwise, the payoff is max(S_T - K, 0).
 *
 * @param s      Initial stock price
 * @param k      Strike price
 * @param b      Barrier level (down-and-out)
 * @param tau    Time to maturity (in years)
 * @param r      Risk-free interest rate
 * @param q      Dividend yield
 * @param v      Volatility (annualized)
 * @param m      Number of time steps in each path
 * @param n      Number of Monte Carlo paths to simulate
 *
 * @return       Expected discounted payoff (option price)
 *
 * @note Uses OpenMP for parallelization over paths.
 */
 
float barrier_omp(float s, float k, float b, float tau, float r, float q, float v, size_t m, size_t n) {
    const float dt = tau/m;
    const float drift = (r - q - v*v/2)*dt;
    const float scale = v*std::sqrt(dt);
    const float logb = std::log(b);
    const float log_s0 = std::log(s);
    float payoffs = 0.0;
	
	#pragma omp parallel for reduction(+:payoffs)
    for (size_t i = 0; i < n; i++) {
        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());
        thread_local std::normal_distribution<float> dist(drift, scale);
        
        float log_price = log_s0;
        bool barrier_hit = false;
        for (size_t j = 0; j < m; j++) {
            log_price += dist(gen);
            if (log_price < logb) {
                barrier_hit = true;
                break;
            }
        }
        if (!barrier_hit) { // Accumulate payoff if barrier not hit
        	payoffs += std::max(std::exp(log_price) - k, 0.f);
        }
    }
    // Calculate discounted expected payoff (option price)
    return std::exp(-r*tau)*payoffs/n;
}


/**
 * @brief Single-threaded helper function for computing path payoffs.
 *
 * This function computes the accumulated payoff for a batch of paths
 *
 * @param log_s0        Initial log price
 * @param k             Strike price
 * @param logb          Log of barrier level
 * @param drift         Drift term for log-price increment
 * @param scale         Volatility scale factor
 * @param m             Number of time steps per path
 * @param n             Number of paths to simulate in this batch
 *
 * @return              Accumulated batch payoffs
 */
float path_payoff(float log_s0, float k, float logb, float drift, float scale, size_t m, size_t n) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> dist(drift, scale);
	float payoff = 0;
	for (size_t i = 0; i<n; i++) {
	  float log_price = log_s0;
	  bool barrier_hit = false;
	  for (size_t j = 0; j < m; j++) {
		log_price += dist(gen);
		if (log_price < logb) {
			barrier_hit = true;
			break;
		}
	  }
	  if (!barrier_hit) { // Accumulate payoff if barrier not hit
	  	payoff += std::max(std::exp(log_price) - k, 0.f);
	  }
	}
	return payoff;
}


/**
 * @brief Multi-threaded Monte Carlo simulation using std::async for parallel path generation.
 *
 * This version splits the workload across hardware threads (using `std::thread::hardware_concurrency`).
 * Each thread runs a batch of path simulations and accumulates payoffs
 *
 * @param s     Initial stock price
 * @param k     Strike price
 * @param b     Barrier level
 * @param tau   Time to maturity
 * @param r     Risk-free rate
 * @param q     Dividend yield
 * @param v     Volatility
 * @param m     Number of time steps per path
 * @param n     Total number of paths to simulate
 *
 * @return      Expected discounted payoff (option price)
 */
float barrier_threads(float s, float k, float b, float tau, float r, float q, float v, size_t m, size_t n) {
    const float dt = tau/m;
    const float drift = (r - q - v*v/2)*dt;
    const float scale = v*std::sqrt(dt);
    const float log_b = std::log(b);
    const float log_s0 = std::log(s);

    float payoffs = 0;
    size_t n_threads = std::thread::hardware_concurrency();
    if (!n_threads) {  // 0 if value could be not well defined or not computable
    	n_threads = 1;
    }
    size_t batch = n/n_threads;
    std::vector<std::future<float>> futures;
    
    for (size_t i = 0; i < n_threads; ++i) { // launch worker threads distributing the load
        futures.emplace_back(std::async(path_payoff, log_s0, k, log_b, drift, scale, m, batch + (i < n%n_threads? 1 : 0)));
	}
    // Wait for all threads to finish
    for (auto& f: futures) {
    	payoffs += f.get();
    }
    // Calculate discounted expected payoff (option price)
    return std::exp(-r*tau)*payoffs/n;
}

// Expose C++ functions to Python
PYBIND11_MODULE(mc, m) {
    m.doc() = "montecarlo simulation c++ extension";
    m.def("barrier_omp", &barrier_omp);
    m.def("barrier", &barrier);
    m.def("barrier_threads", &barrier_threads);
}
