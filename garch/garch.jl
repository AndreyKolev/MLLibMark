"GARCH LLH benchmark, Julia version."

import JSON
using Test
using Statistics

data = JSON.parsefile("data.json")
out = Dict()

function garchSim(ɛ²::Vector, ω, α, β)
    h = similar(ɛ²)
    h[1] = mean(ɛ²)
    for i = 2:length(ɛ²)
        @inbounds h[i] = ω + α*ɛ²[i-1] + β*h[i-1]
    end
    h
end

function garchLLH(y::Vector, params::Vector)
    ɛ² = y.^2
    T = length(y)
    h = garchSim(ɛ², params...)
    -(T-1)*log(2π)/2 - sum(log.(h) + (y./sqrt.(h)).^2)/2
end

function benchmark(n)
    #price = Array{Float32}(data["price"])
    ret = Array{Float32}(data["ret"])
    #ret = diff(log.(price))
    #ret = ret .- mean(ret)
    x0 = Array{Float32}(data["x0"])
    tmp = garchLLH(ret, x0)  # warm-up
    @test tmp ≈ data["llh"] atol=1e-3
    t = @elapsed for i = 1:n; tmp = garchLLH(ret, x0); end
end

n_ix = findfirst(ARGS.=="-n")
if n_ix < length(ARGS)
    n = tryparse(Int64, ARGS[n_ix+1])
    n==nothing && error("Wrong number of iterations!")
else
    error("Number of iterations is not set!")
end

out["julia"] = benchmark(n)
println(JSON.json(out))