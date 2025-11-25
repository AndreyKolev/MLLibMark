#' MC Barrier Option pricing benchmark, R version.
library(jsonlite)
library(parallel)

# Determine execution mode: "std" (sequential) or "parallel"
# Mode can be overridden via command-line argument: -mode parallel
mode = "std"
args = commandArgs(trailingOnly = T)
if (length(args) == 2 && args[1] == '-mode') mode = args[2]

# Monte Carlo pricing of a down-and-out barrier option
# S0: Initial stock price
# K: Strike price
# B: Barrier level (down-and-out)
# tau: Time to maturity
# r: Risk-free interest rate
# q: Dividend yield
# v: Volatility
# M: Number of time steps (discretization)
# N: Number of Monte Carlo simulations (replications)
barrier <- function(S0, K, B, tau, r, q, v, M, N) {
  # Time step size
  dt <- tau/M
  # Drift term
  drift <- (r - q - v*v/2)*dt
  # Volatility scaling for Brownian motion
  scale <- v*sqrt(dt)
  # Inner function to simulate one path and compute payoff
  payoff <- function() {
    S <- log(S0) + cumsum(rnorm(M, drift, scale))
    ifelse(min(S) <= log(B), 0, max(exp(tail(S, 1)) - K, 0))
  }
  # Choose execution strategy based on mode
  if (mode == "parallel") {
  	num.cores <- detectCores()
    payoffs <- unlist(mclapply(1:N, \(x) payoff(), mc.cores = num.cores))
  } else {
    payoffs <- replicate(N, payoff())
  }
  # Discount the average payoff back to present value
  exp(-r*tau)*mean(payoffs)
}

# Read input data from JSON file
data = read_json("data.json")
# Warm up
call_price <- with(data, barrier(price, strike, barrier, tau, rate, dy, vol, time_steps, 1))
t <- Sys.time()
call_price <- with(data, barrier(price, strike, barrier, tau, rate, dy, vol, time_steps, n_rep))
# Measure total execution time
t <- as.double(difftime(Sys.time(), t, units = "secs"))
# Ensure the computed result matches the expected benchmark value within tolerance.
stopifnot(all.equal(call_price, data$val, tolerance = data$tol))
# Output: Report execution time in JSON format
cat(paste0("{\"R-", mode, "\": ", t, "}\n"))
