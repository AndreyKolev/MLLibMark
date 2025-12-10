# ==============================================================================
# Hamiltonian Monte Carlo for Bayesian Logistic Regression Benchmark
# 
# This file implements Hamiltonian Monte Carlo (HMC) sampling for Bayesian
# logistic regression using the a9a (Adult) dataset.
# 
# Author: Andrey Kolev
# ==============================================================================


using Statistics
using PyCall
using JSON

datasets = pyimport("sklearn.datasets")


"""
    σ(x::T) where {T <: AbstractFloat}

Compute the sigmoid (logistic) function: `σ(x) = 1 / (1 + exp(-x))`.
"""
σ(x::T) where {T <: AbstractFloat} = 1 ./(1 .+ exp(-x))


"""
    softplus(x::T) where {T <: AbstractFloat}

Numerically stable implementation of the softplus function: `log(1 + exp(x))`.
"""
softplus(x::T) where {T <: AbstractFloat} =  max(x, 0) + log1p(exp(-abs(x)))

"""
    get_data()

Download and load the a9a (Adult) classification dataset from the LIBSVM dataset collection for binary classification.
"""
function get_data()
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    train_file = "a9a"
    test_file = train_file * ".t"
    !isfile(train_file) && download(url * train_file, train_file)
    !isfile(test_file) && download(url * test_file, test_file)
    X_train, y_train = datasets.load_svmlight_file(train_file)
    X_train = X_train.todense()
    X_test, y_test = datasets.load_svmlight_file(test_file)
    X_test = hcat(X_test.todense(), zeros(eltype(X_train), size(y_test)[1]))
    
    y_train = ((y_train .+ 1)/2)
    y_test = ((y_test .+ 1)/2)
    Array{Float32}(X_train), Array{Float32}(y_train), Array{Float32}(X_test), Array{Float32}(y_test)
end


"""
    hmc_step(U::Function, ∇U::Function, ϵ::T, L::Integer, current_q::Array{T}) where {T <: AbstractFloat}

Perform a single HMC step using the leapfrog integrator.
"""
function hmc_step(U::Function, ∇U::Function, ϵ::T, L::Integer, current_q::Array{T}) where {T <: AbstractFloat}
    q = copy(current_q)
    # Sample momentum
    p = randn(T, length(q))
    current_p = p
    # half step for momentum
    p -= ϵ*∇U(q)/2
    @simd for i in 1:L
         # full step for the position
        q += ϵ*p
        # full step for the momentum, except at end of trajectory
        if i < L
            p -= ϵ*∇U(q)
        end
    end
    # Final half-step momentum update (negated for symmetry)
    p = ϵ*∇U(q)/2 - p
    # Compute potential and kinetic energies at current and proposed states
    current_u = U(current_q)
    current_k = (current_p'*current_p)/2
    proposed_u = U(q)
    proposed_k = (p'*p)/2
    # Compute acceptance probability
    rand() < exp.(current_u .- proposed_u .+ current_k .- proposed_k)[1] ? q : current_q
end


"""
    hmc(x::Array{T}, y::Array{T}, ϵ::T, L::Integer, α::T, n_iter::Integer) where {T <: AbstractFloat} 

Run Hamiltonian Monte Carlo sampling for Bayesian logistic regression.
"""
function hmc(x::Array{T}, y::Array{T}, ϵ::T, L::Integer, α::T, n_iter::Integer) where {T <: AbstractFloat} 
    # Define negative log-posterior (potential energy)
    U(β) = sum(softplus.(x*β)) .- y'*(x*β) .+ (β'*β)/2α
    # Gradient of the energy function
    ∇U(β) = x'*(σ.(x*β) - y) .+ β/α
    # Initialize starting point
    q = zeros(T, size(x, 2), 1)
    # Run HMC iterations
    samples = zeros(T, n_iter, size(x, 2))    
    for i in 1:n_iter
        q = hmc_step(U, ∇U, ϵ, L, q)
        samples[i, :] = q
    end
    samples
end

function benchmark()
"""
    benchmark()

Run a HMC benchmark on the a9a (Adult) dataset.
"""
    # Load data
    x_train, y_train, x_test, y_test = get_data()
    # Load hyperparameters
    params = JSON.parsefile("params.json")
    α = Float32(params["alpha"])
    n_iter = Int(params["n_iter"])
    ϵ = Float32(params["epsilon"])
    n_leaps = Int(params["n_leaps"])
    burn_in =  Int(params["burn_in"]) 
    # Warm up run
    hmc(x_train, y_train, ϵ, n_leaps, α, 1)
    # Perform full HMC sampling
    runtime = @elapsed samples = hmc(x_train, y_train, ϵ, n_leaps, α, n_iter)
    # Estimate posterior mean after burn-in
    posterior_mean = mean(samples[burn_in:end, :], dims=1)
    # Calculate accuracy on test set
    accuracy = mean(float(σ.((x_test*posterior_mean')) .> 0.5) .== y_test)
    # Validate accuracy
    @assert accuracy > params["val_accuracy"]
    runtime
end

println(JSON.json(Dict("julia" => benchmark())))
