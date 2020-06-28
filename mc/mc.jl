import JSON
using Test
using Distributed
using Statistics

arg_mode = "std"
if length(ARGS)>1 && ARGS[1]=="-mode"
    if ARGS[2] ∈ ["matrix", "parallel"]
        addprocs()
        arg_mode = ARGS[2]
    elseif ARGS[2]≠"std"
        error("-mode: invalid choice: $(ARGS[2]) (choose from 'std', 'matrix', 'parallel')")
    end
else
    error("use -mode MODE and choose MODE from 'std', 'matrix', 'parallel'")
end

data = JSON.parsefile("data.json")

"Generate GBM price paths"
function paths(S, tau, r, q, v, M::Int, N::Int)
	dt = tau/M
    S*exp.(cumsum((r-q-v/2)dt .+ √(v*dt)randn(Float32, M, N), dims=1))
end

"Price a barrier option (using path matrix)"
function barrier_mat(S, K, B, tau, r, q, v, M::Int, N::Int)
	S = paths(S, tau, r, q, v, M, N)
	l = mapslices(x->all(x.>B), S, dims=1)
    payoffs = l.*max.(S[end, :].-K, 0)'
	exp(-r*tau)*mean(payoffs)
end

"Price a barrier option"
function barrier(S, K, B, tau, r, q, v, M::Int, N::Int)
    dt = tau/M
    g1 = (r-q-v/2)*dt
    g2 = sqrt(v*dt)

    payoffs = @distributed (+) for i in 1:N
        cumsum = log(S)
        min_price = Inf32
        for j in 1:M
            cumsum += g1+g2*randn(Float32)
            price = exp(cumsum)
            if price < min_price
                min_price = price
            end
        end
        min_price>B ? max(exp(cumsum)-K, 0) : 0
    end
    exp(-r*tau)*payoffs/N
end

"Benchmark barrier option pricing"
function benchmark(mode::String)
    fun = (mode=="matrix") ? barrier_mat : barrier 
    call_price() = fun(data["price"], data["strike"],
              data["barrier"], data["tau"], data["rate"], data["dy"],
              data["vol"], data["time_steps"], data["n_rep"])
    @test call_price() ≈ data["val"] atol=data["tol"]  #warm-up and validate output
    @elapsed call_price()
end

println(JSON.json(Dict("julia-$arg_mode" => benchmark(arg_mode))))
