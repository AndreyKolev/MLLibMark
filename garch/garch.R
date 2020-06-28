#' GARCH LLH benchmark, R version
library(jsonlite)

data <- read_json("data.json")
val.llh <- data$llh
x0 <- as.numeric(data$x0)
ret <- as.numeric(data$ret)
#price <- as.numeric(data$price)
#ret <- diff(log(price))
#ret <- ret-mean(ret)

args <- paste(commandArgs(trailingOnly=T), collapse=' ')
argm <- unlist(regmatches(args, gregexpr("-([[:alpha:]]+)[[:blank:]]([[:alnum:]]+)", args)))
arg <- setNames(as.list(sapply(strsplit(argm, " "), '[', 2)), sapply(strsplit(argm, " "), '[', 1))
mode = arg[["-mode"]]
n = as.numeric(arg[["-n"]])

garch.sim <- function(ret.2, par){
  Reduce(f=function(ht, y) par[1]+par[2]*y+par[3]*ht,
         x=ret.2, 
         init=mean(ret.2),
         accumulate=T)
}
garch.LLH <- function(ret, par){
  h <- head(garch.sim(ret^2, par), -1)
  t <- length(ret)
  -0.5*(t-1)*log(2*pi)-0.5*sum(log(h)+(ret/sqrt(h))^2)
}

llh <- garch.LLH(ret, x0)
stopifnot(all.equal(llh, val.llh, tolerance=1e-3))
t <- Sys.time()
for(i in 1:n) llh <- garch.LLH(ret, x0)
t <- as.double(difftime(Sys.time(), t, units="secs"))
cat(paste0("{\"R-", mode, "\": ", t, "}\n"))