# ==============================================================================
# Barrier Option Pricing using Monte Carlo Simulation in Julia
# 
# This script implements multiple Monte Carlo methods for pricing barrier options,
# including sequential, multi-threaded, distributed, and matrix-based approaches.
# 
# Author: Andrey Kolev
# ==============================================================================

import JSON
using Test
using Distributed
using Statistics

# Load input parameters from data.json
data = JSON.parsefile("data.json")


"""
    paths(s::T, tau::T, r::T, q::T, v::T, m::Int, n::Int)

Generate log-price paths of a Geometric Brownian Motion (GBM)
"""
function paths(s::T, tau::T, r::T, q::T, v::T, m::Int, n::Int) where T<:AbstractFloat
	dt = tau/m
    log(s) .+ cumsum((r - q - v*v/2)dt .+ v*√(dt)randn(T, m, n), dims=1)
end


"""
    barrier_mat(s0::T, k::T, b::T, tau::T, r::T, q::T, v::T, m::Int, n::Int)

Price a barrier option using vectorized Monte Carlo approach.
"""
function barrier_mat(s0::T, k::T, b::T, tau::T, r::T, q::T, v::T, m::Int, n::Int) where T<:AbstractFloat
	s = paths(s0, tau, r, q, v, m, n)
	payoffs = (.~any(s .< log(b), dims=1)).*max.(exp.(s[end,:]) .- k, 0)'
	exp(-r*tau)*mean(payoffs)
end


"""
    barrier_distributed(s::T, k::T, b::T, tau::T, r::T, q::T, v::T, m::Int, n::Int)

Distributed Monte Carlo pricing using Julia's Distributed module.
"""
function barrier_distributed(s::T, k::T, b::T, tau::T, r::T, q::T, v::T, m::Int, n::Int) where T<:AbstractFloat
    dt = tau/m
    drift = (r - q - v*v/2)*dt
    scale = v*√(dt)
    logb = log(b)
    log_s0 = log(s)
    payoffs = @distributed (+) for i in 1:n
        cumsum = log_s0
        min_price = T(Inf) #convert(T, Inf)
        for j in 1:m
            cumsum += drift + scale*randn(T)
            (cumsum < min_price) && (min_price = cumsum)
        end
        min_price > logb ? max(exp(cumsum) - k, T(0)) : T(0)
    end
    exp(-r*tau)*payoffs/n
end


"""
    paths_batch(log_s0::T, k::T, logb::T, drift::T, scale::T, m::Int, n::Int)

Compute accumulated payoff for a batch of paths in a single thread.
"""
function paths_batch(log_s0::T, k::T, logb::T, drift::T, scale::T, m::Int, n::Int) where T<:AbstractFloat
	payoff = T(0)
	for i ∈ 1:n
		log_price = log_s0
		barrier_hit = false
		for j ∈ 1:m
			log_price += drift + scale*randn(T)
			if log_price < logb
				barrier_hit = true
				break
			end
		end
		if !barrier_hit
			payoff += max(exp(log_price) - k, T(0))
		end
	end
	payoff
end


"""
    barrier_threads(s::T, k::T, b::T, tau::T, r::T, q::T, v::T, m::Int, n::Int)

Multi-threaded Monte Carlo barrier option pricing using Julia's Threads module.
"""
function barrier_threads(s::T, k::T, b::T, tau::T, r::T, q::T, v::T, m::Int, n::Int) where T<:AbstractFloat
    dt = tau/m
    drift = (r - q - v*v/2)*dt
    scale = v*√(dt)
    logb = log(b)
    log_s0 = log(s)
    n_threads = max(Threads.nthreads(), 1)
    batch = div(n, n_threads) 
	tasks = map(1:n_threads) do i
		ni = batch + (i < n%n_threads ? 1 : 0)
		Threads.@spawn paths_batch(log_s0, k, logb, drift, scale, m, ni)
	end
    payoffs = fetch.(tasks)
    exp(-r*tau)*sum(payoffs)/n
end


"""
    barrier(s::T, k::T, b::T, tau::T, r::T, q::T, v::T, m::Int, n::Int)

Sequential Monte Carlo pricing — the baseline method.
"""
function barrier(s::T, k::T, b::T, tau::T, r::T, q::T, v::T, m::Int, n::Int) where T<:AbstractFloat
	dt = tau/m
    drift = (r - q - v*v/2)*dt
    scale = v*√(dt)
    logb = log(b)
    log_s0 = log(s)
	payoff = T(0)
	for i ∈ 1:n
		log_price = log_s0
		barrier_hit = false
		for j ∈ 1:m
			log_price += drift + scale*randn(T)
			if log_price < logb
				barrier_hit = true
				break
			end
		end
		if !barrier_hit
			payoff += max(exp(log_price) - k, T(0))
		end
	end
	exp(-r*tau)*payoff/n
end

# Map execution modes to their respective functions
modes = Dict("std" => barrier, "matrix" => barrier_mat, "distributed" => barrier_distributed, "threads" => barrier_threads)
if length(ARGS)>1 && ARGS[1] == "-mode"
    if ARGS[2] ∈ keys(modes)
    	arg_mode = ARGS[2]
    	(ARGS[2] == "distributed") && addprocs()
    elseif ARGS[2] ≠ "std"
        error("-mode: invalid choice: $(ARGS[2]) (choose from $(keys(modes)))")
    end
else
    error("use -mode MODE and choose MODE from $(keys(modes))")
end

"""
    benchmark(mode::String, T::Type=Float32)

Run a performance benchmark using the specified pricing method.
"""
function benchmark(mode::String, T::Type=Float32)
    fun = modes[mode]
    call_price() = fun(T(data["price"]), T(data["strike"]),
              T(data["barrier"]), T(data["tau"]), T(data["rate"]), T(data["dy"]),
              T(data["vol"]), data["time_steps"], data["n_rep"])
    @test call_price() ≈ data["val"] atol=data["tol"]  #warm-up and validate output
    @elapsed call_price()
end

# Run benchmark and output result in JSON format
println(JSON.json(Dict("julia-$arg_mode" => benchmark(arg_mode, Float32))))
