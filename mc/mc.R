#' MC Barrier Option pricing benchmark, R version.
library(jsonlite)
library(parallel)

mode = "std"
args = commandArgs(trailingOnly=T)
if (length(args)==2 && args[1]=='-mode') mode = args[2]

num.cores <- detectCores()

barrier <- function(S0, K, B, tau, r, q, v, M, N){
  #' Price a barrier option
  dt <- tau/M
  g1 <- (r-q-v/2)*dt
  g2 <- sqrt(v*dt)
  
  payoff <- function(){
    S <- S0*exp(cumsum(g1 + g2*rnorm(M)))
    l <- min(S) > B
    l*max(tail(S, 1)-K, 0)
  }
  if(mode=="parallel") payoffs <- unlist(mclapply(1:N, function(x) payoff(), mc.cores=num.cores))
  else payoffs <- unlist(lapply(1:N, function(x) payoff()))
  exp(-r*tau)*mean(payoffs)
}

data = read_json("data.json")
call_price <- with(data, barrier(price, strike, barrier, tau, rate, dy, vol, time_steps, 1))
t <- Sys.time()
call_price <- with(data, barrier(price, strike, barrier, tau, rate, dy, vol, time_steps, n_rep))
t <- as.double(difftime(Sys.time(), t, units="secs"))
stopifnot(all.equal(call_price, data$val, tolerance=data$tol)) # validate result
cat(paste0("{\"R-", mode, "\": ", t, "}\n"))