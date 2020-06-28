library(sparsio)
library(jsonlite)

sigmoid <- function(x){1/(1+exp(-x))}

get.data <- function() {
    url <- "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    train.file <- "a9a"
    test.file <- paste0(train.file, ".t")
    if(!file.exists(train.file)){
        download.file(url=paste0(url, train.file), destfile=train.file, quiet=T)
    }
    if(!file.exists(test.file)){
        download.file(url=paste0(url, test.file), destfile= test.file, quiet=T)
    }
    train.data <- sparsio::read_svmlight(train.file)
    test.data <- sparsio::read_svmlight(test.file)
    list(X.train=as.matrix(train.data$x),
         y.train=(train.data$y+1)/2,
         X.test=cbind(as.matrix(test.data$x), rep(0, nrow(test.data$x))),
         y.test = (test.data$y+1)/2)
}

hmc <- function(U, dU, epsilon, L, current_q){
    q <- current_q
    p <- rnorm(length(q))
    current_p <- p
    p <- p - 0.5*epsilon*dU(q)
    for(i in 1:L){
        q <- q + epsilon*p
        if (i != L){
            p <- p-epsilon*dU(q)
        }
    }
    p <- p - 0.5*epsilon*dU(q)
    p <- -p
    current_U <- U(current_q)
    current_K <- 0.5*(sum(current_p^2))
    proposed_U <- U(q)
    proposed_K <- 0.5*(sum(p^2))
    if (log(runif(1)) < (current_U-proposed_U+current_K-proposed_K)) q else current_q
}

lr_hmc <- function(y, X, epsilon, L, alpha, n_iter) {
    U <- function(beta){sum(log(1+exp(X%*%beta))) - y%*%(X%*%beta) + (0.5/alpha)*sum(beta^2)}
    dU <- function(beta){t(X)%*%(exp(X%*%beta)/(1+exp(X%*%beta))-y) + beta/alpha }
    D <- ncol(X)
    q <- rep(0, ncol(X))
    out <- matrix(rep(0, n_iter*D), n_iter, D)
    for(i in 1:n_iter){
        q <- hmc(U, dU, epsilon, L, q)
        out[i,] <- q
    }
    out
}

params <- jsonlite::read_json("params.json")
data <- get.data()
w <- lr_hmc(data$y.train, data$X.train, params$epsilon, params$n_leaps, params$alpha, 1)
t <- Sys.time()
w <- lr_hmc(data$y.train, data$X.train, params$epsilon, params$n_leaps, params$alpha, params$n_iter)
t <- as.double(difftime(Sys.time(), t, units="secs"))
coef <- apply(head(w, -params$burn_in), 2, mean)
acc <- mean((sigmoid(data$X.test%*%coef)>0.5) == data$y.test)
stopifnot(acc>0.8)
cat(paste0("{\"R\": ", t, "}\n"))