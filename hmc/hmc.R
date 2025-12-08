# =============================================================================
# Logistic Regression with Hamiltonian Monte Carlo (HMC)
#
# This script implements Hamiltonian Monte Carlo (HMC) for logistic regression
# using sparse data input via the SVMlight format. It downloads the a9a dataset,
# performs Bayesian inference on the logistic regression coefficients via HMC,
# and evaluates the model's accuracy on the test set.
#
# The implementation uses:
#   - sparsio: for reading SVMlight-format files
#   - jsonlite: for loading hyperparameters from JSON
#   - Custom HMC sampler with leapfrog integration
#
# Author: Andrey Kolev
# =============================================================================

library(sparsio)
library(jsonlite)

# Numerically stable log(1 + exp(x)) 
log1p.exp <- function(x) {
  pmax(x, 0) + log1p(exp(-abs(x)))
}

# =============================================================================
# Data Loading Function
# =============================================================================
#' Download and load the a9a dataset from LIBSVM tools
#'
#' This function downloads the a9a training and test datasets from the official
#' LIBSVM dataset repository if they are not already present locally.
#' The data is read using sparsio::read_svmlight and converted into dense matrices.
#' The target variable is transformed from {-1, 1} to {0, 1} for binary classification.
#'
#' @return A list containing:
#'   - X.train: training feature matrix (n x p)
#'   - y.train: training labels (0 or 1)
#'   - X.test: test feature matrix with an extra column of zeros (for consistency)
#'   - y.test: test labels (0 or 1)
get.data <- function() {
    url <- "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    train.file <- "a9a"
    test.file <- paste0(train.file, ".t")
    # Download train and test file if missing
    if (!file.exists(train.file)) {
        download.file(url = paste0(url, train.file), destfile = train.file, quiet = T)
    }
    if (!file.exists(test.file)) {
        download.file(url = paste0(url, test.file), destfile = test.file, quiet = T)
    }
    # Read SVMlight files
    train.data <- sparsio::read_svmlight(train.file)
    test.data <- sparsio::read_svmlight(test.file)
    # Convert to dense matrices and transform labels: y ∈ {-1, 1} → {0, 1}
    list(X.train = as.matrix(train.data$x),
         y.train = (train.data$y + 1)/2,
         X.test = cbind(as.matrix(test.data$x), rep(0, nrow(test.data$x))),
         y.test = (test.data$y + 1)/2)
}


# =============================================================================
# Hamiltonian Monte Carlo (HMC) Sampler
# =============================================================================

#' Hamiltonian Monte Carlo (HMC) Step
#'
#' Performs a single HMC update step using the leapfrog integrator.
#'
#' @param U function: negative log-posterior (potential energy)
#' @param dU function: gradient of U (negative log-posterior)
#' @param epsilon numeric: step size (learning rate) for leapfrog
#' @param L integer: number of leapfrog steps
#' @param current_q numeric vector: current position (e.g., regression coefficients)
#'
#' @return updated position (beta) after HMC proposal or current position if rejected
hmc <- function(U, dU, epsilon, L, current_q){
    q <- current_q
    p <- rnorm(length(q))  # Sample momentum
    current_p <- p
    # Half-step momentum update
    p <- p - 0.5*epsilon*dU(q)
    # Leapfrog integrator
    for (i in 1:L) {
        q <- q + epsilon*p
        if (i != L) {
            p <- p - epsilon*dU(q)
        }
    }
    # Final half-step momentum update (negated for symmetry)
    p <- 0.5*epsilon*dU(q) - p
    # Compute Hamiltonian values at current and proposed states
    current_U <- U(current_q)
    current_K <- 0.5*(sum(current_p^2))
    proposed_U <- U(q)
    proposed_K <- 0.5*(sum(p^2))
    # Metropolis acceptance
    if (log(runif(1)) < (current_U - proposed_U + current_K - proposed_K)) {
      q  # Accept proposal
    } else {
      current_q  # Reject and return current state
    }
}

# =============================================================================
# Logistic Regression HMC Sampler
# =============================================================================

#' Hamiltonian Monte Carlo for Logistic Regression
#'
#' Performs Bayesian inference on logistic regression coefficients using HMC.
#' The posterior is approximated with a Gaussian prior (regularization via alpha).
#'
#' @param y numeric vector: binary response variable (0 or 1)
#' @param X matrix: feature matrix of size n x p
#' @param epsilon numeric: step size for leapfrog integration
#' @param L integer: number of leapfrog steps per HMC iteration
#' @param alpha numeric: inverse variance of the Gaussian prior (controls regularization)
#' @param n_iter integer: total number of HMC iterations (including burn-in)
#'
#' @return matrix of size n_iter x p: sampled beta coefficients
lr_hmc <- function(y, X, epsilon, L, alpha, n_iter) {
    # Define negative log-posterior (potential energy)  
    U <- \(beta) {
      X.beta <- X %*% beta
      sum(log1p.exp(X.beta)) - y %*% X.beta + sum(beta^2)/(2*alpha)
    }
    # Define gradient of U (dU/dbeta)
    dU <- \(beta) {t(X) %*% (plogis(X %*% beta) - y) + beta/alpha}
    # Initialize starting point
    state <- rep(0, ncol(X))
    # Storage for results
    samples <- matrix(0, n_iter, ncol(X))
    # Run HMC iterations
    for (i in 1:n_iter) {
        state <- hmc(U, dU, epsilon, L, state)
        samples[i,] <- state
    }
    samples
}
# Load parameters from JSON file
params <- jsonlite::read_json("params.json")
params$burn_in <- 10
params$n_iter <- 50
data <- get.data()
# warm up run
lr_hmc(data$y.train, data$X.train, params$epsilon, params$n_leaps, params$alpha, 1)
# Perform full HMC sampling
elapsed <- Sys.time()
samples <- lr_hmc(data$y.train, data$X.train, params$epsilon, params$n_leaps, params$alpha, params$n_iter)
# elapsed time
elapsed <- as.double(difftime(Sys.time(), elapsed, units = "secs"))
# Estimate posterior mean after burn-in
posterior.mean <- apply(head(samples, -params$burn_in), 2, mean)
# Predict on test set
accuracy <- mean((plogis(data$X.test %*% posterior.mean) > 0.5) == data$y.test)
# Validate accuracy (must be > 80%)
stopifnot(accuracy > 0.8)
cat(paste0("{\"R\": ", elapsed, "}\n"))
