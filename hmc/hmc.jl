using Statistics
using PyCall
using JSON

datasets = pyimport("sklearn.datasets")

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

function hmc(U, ∇U, ϵ, L, current_q)
    q = current_q
    p = randn(Float32, length(q))
    current_p = p
    # half step for momentum
    p = p - ϵ*∇U(q)/2
    @simd for i in 1:L
        # full step for the pos
        q = q + ϵ*p
        # full step for the momentum
        if i != L
            p = p - ϵ*∇U(q)
        end
    end
    # half step for momentum
    p = p - ϵ*∇U(q)/2
    # Negate momentum for symmetry
    p = -p
    current_U = U(current_q)
    current_K = (current_p'*current_p)/2
    proposed_U = U(q)
    proposed_K = (p'*p)/2
    # Accept or reject
    rand() < exp.(current_U .- proposed_U .+ current_K .- proposed_K)[1] ? q : current_q
end

function lr_hmc(X, y, ϵ, L, α, n_iter)
    U(β) = sum(log.(1 .+ exp.(X*β))) .- y'*(X*β) .+ (β'*β)/2α
    ∇U(β) = X'*(exp.(X*β)./(1 .+ exp.(X*β))-y).+ β/α
    D = size(X)[2]
    let q = zeros(Float32, size(X)[2], 1)
    hcat([q = hmc(U, ∇U, ϵ, L, q) for _ in 1:n_iter]...)'
    end
end

function benchmark()
    sigmoid(x) = 1 ./(1 .+ exp.(-x))
    X_train, y_train, X_test, y_test = get_data()
    params = JSON.parsefile("params.json")
    α = Float32(params["alpha"])
    n_iter = Int(params["n_iter"])
    ϵ = Float32(params["epsilon"])
    n_leaps = Int(params["n_leaps"])
    burn_in =  Int(params["burn_in"]) 
    lr_hmc(X_train, y_train, ϵ, n_leaps, α, 1)
    t = @elapsed z = lr_hmc(X_train, y_train, ϵ, n_leaps, α, n_iter)
    coef = mean(z[burn_in:end, :], dims=1)
    acc = mean(float(sigmoid((X_test*coef')) .> 0.5) .== y_test)
    @assert acc > 0.8
    t
end

println(JSON.json(Dict("julia" => benchmark())))