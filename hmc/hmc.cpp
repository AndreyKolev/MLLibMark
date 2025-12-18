#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <print>
#include <expected>
#include <random>
#include <format>
#include <system_error>
#include <string_view>
#include <ranges>
#include <list>
#include <map>
#include <charconv>
#include <Eigen/Dense>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace std::literals;
using Eigen::MatrixXf;
using json = nlohmann::json;

/**
 * @brief Computes the sigmoid activation function element-wise.
 *
 * Applies the logistic function: σ(z) = 1 / (1 + exp(-z)) to each element of the input matrix.
 *
 * @param x Input matrix of floats.
 * @return MatrixXf containing sigmoid of each element.
 */
MatrixXf sigmoid(const MatrixXf& x) {
    return 1.0f/(1.0f + (-x).array().exp());
}

/**
 * @brief Computes the numerically stable softplus function element-wise: max(0, x) + log(1 + exp(-|x|)).
 *
 * Softplus function: log(1 + exp(x))
 *
 * @param x Input matrix of floats.
 * @return MatrixXf containing softplus of each element.
 */
MatrixXf softplus(const MatrixXf& x) {
    return x.array().max(0.f) + (-x.array().abs()).exp().log1p();
}


/**
 * @brief Computes the negative log-posterior (potential energy) U(beta | y, x, alpha).
 *
 * This is the core objective function in Bayesian logistic regression:
 *   U = sum_i[softplus(x_i^T β)] - y^T (X β) + ||β||² / (2α)
 *
 * Where:
 *   - y: target vector (binary labels)
 *   - x: design matrix (features)
 *   - β: coefficients to estimate
 *   - α: inverse of variance (precision) of prior on β
 *
 * @param y Target vector (Nx1).
 * @param x Feature matrix (NxM).
 * @param beta Coefficient vector (Mx1).
 * @param alpha Precision parameter (inverse of variance) of the prior on beta.
 * @return Scalar value of potential energy.
 */
float u(const MatrixXf& y, const MatrixXf& x, const MatrixXf& beta, float alpha) {
    auto x_beta = x*beta;
    return softplus(x_beta).sum() - (y.transpose()*x_beta)(0,0) + (beta.transpose()*beta)(0,0)/(2*alpha);
}

/**
 * @brief Computes the gradient of the potential energy ∇_β U with respect to β.
 *
 * The gradient is:
 *   ∇β U = X^T (σ(Xβ) - y) + β / α
 *
 * Where σ is the sigmoid function.
 *
 * @param y Target vector (Nx1).
 * @param x Feature matrix (NxM).
 * @param beta Coefficient vector (Mx1).
 * @param alpha Precision parameter for the prior on β.
 * @return Gradient vector (Mx1).
 */
MatrixXf grad_u(const MatrixXf& y, const MatrixXf& x, const MatrixXf& beta, float alpha) {
    return x.transpose()*(sigmoid(x*beta) - y) + beta/alpha;
}


/**
 * @brief Performs a single Hamiltonian Monte Carlo (HMC) step.
 *
 * Implements the leapfrog integrator.
 *
 * @param y Target vector (Nx1).
 * @param x Feature matrix (NxM).
 * @param epsilon Step size for the leapfrog integrator.
 * @param leapfrog_steps Number of leapfrog steps per HMC update.
 * @param current_q Current state (beta vector).
 * @param alpha Precision of prior on beta.
 * @return Updated beta vector (accepted or rejected).
 */
MatrixXf hmc_step(const MatrixXf& y, const MatrixXf& x, float epsilon, size_t leapfrog_steps, const MatrixXf& current_q, float alpha) {
    // Initialize random engine and distributions
    std::mt19937_64 engine(std::random_device{}());
    std::normal_distribution<float> dist(0.0, 1.0);
    std::uniform_real_distribution<float> uniform;
    
    // Initialize momentum p and current state q
    MatrixXf p = MatrixXf::NullaryExpr(current_q.rows(), current_q.cols(), [&](){return dist(engine);});
    MatrixXf current_p = p;
    MatrixXf q = current_q;
    
    // Half-step momentum update
    p = p - 0.5f*epsilon*grad_u(y, x, q, alpha);
    for (size_t i=0; i < leapfrog_steps; ++i) {
        // Position update
        q = q + epsilon*p;
        // Momentum update (except last step)
        if (i < leapfrog_steps - 1) {
            p = p - epsilon*grad_u(y, x, q, alpha);
        }
    }
    // Final half-step momentum update (negated for symmetry)
    p = epsilon*grad_u(y, x, q, alpha)/2 - p;
    // Compute potential and kinetic energies at current and proposed states
    auto current_u = u(y, x, current_q, alpha);
    auto current_k = (current_p.transpose()*current_p)(0,0)/2.f;
    auto proposed_u = u(y, x, q, alpha);
    auto proposed_k = (p.transpose()*p)(0,0)/2.f;
    // Acceptance
    return uniform(engine) < std::exp(current_u - proposed_u + current_k - proposed_k) ? q : current_q;
}

/**
 * @brief Runs Hamiltonian Monte Carlo (HMC) sampler to generate posterior samples.
 *
 * @param y Target vector (Nx1).
 * @param x Feature matrix (NxM).
 * @param epsilon Step size for leapfrog integrator.
 * @param leapfrog_steps Number of leapfrog steps per HMC iteration.
 * @param alpha Precision of prior on beta.
 * @param n_iter Number of total HMC iterations.
 * @return Matrix of samples: n_iter × M, where each row is a β sample.
 */
MatrixXf hmc(const MatrixXf& y, const MatrixXf& x, float epsilon, size_t leapfrog_steps, float alpha, size_t n_iter){
    // Initialize starting point
    MatrixXf q = MatrixXf::Zero(x.cols(), 1);
    
    // Pre-allocate storage for samples
    MatrixXf samples = MatrixXf::Zero(n_iter, x.cols());
    
    // Run HMC iterations
    for (size_t i=0; i < n_iter; ++i) {
        q = hmc_step(y, x, epsilon, leapfrog_steps, q, alpha);
        samples.row(i) = q.transpose();
    }
    return samples;
}

/**
 * @brief Reads a SVM-light formatted dataset from file.
 *
 * Supports sparse format: `label index:value index:value ...`
 * Converts sparse features to dense matrix;
 *
 * @param path Path to the SVM-light file.
 * @param cols Optional number of columns to use. If not provided, infers from data.
 * @return std::expected<std::pair<MatrixXf, MatrixXf>, std::string>: success with (X, y), or error message.
 */
auto read_svm_light(const std::filesystem::path& path, std::optional<size_t> cols=std::nullopt)
  -> std::expected<std::pair<MatrixXf, MatrixXf>, std::string> {
    if (!std::filesystem::exists(path)) {
        return std::unexpected("no such file or directory: " + path.string());
    }
    std::ifstream file(path);
    if (!file.is_open()){
        return std::unexpected("unable to open the file: " + path.string());
    }
    std::string_view delim = " ";
    std::string line;
    size_t line_num = 0;
    size_t n_cols = 0;
    std::list<float> yl;
    std::list<std::map<size_t, float>> x_sparse;
    while (std::getline(file, line)) {
        if (line.empty() || line.starts_with('#')) {
            continue;
        }
        std::map<size_t, float> row;
        for (auto [ix, field] : std::views::enumerate(std::views::split(line, delim))) {
            auto view = std::string_view{field.begin(), field.end()};
            if (!view.size()) {
                continue;
            }
            if (!ix) {
                float yf;
                if (std::from_chars(view.starts_with('+') ? view.data() + 1 : view.data(), view.data() + view.size(), yf).ec != std::errc{}){
                    return std::unexpected(std::format("bad label '{}' at line {}", view, line_num));
                }
                yl.emplace_back(yf > 0 ? yf : 0);

            } else {
                auto colon = view.find(':');
                if (colon == std::string_view::npos) {
                    return std::unexpected(std::format("malformed line {}", line_num));
                }
                std::string_view ix_view = view.substr(0, colon);
                std::string_view value_view = view.substr(colon + 1);
                size_t ix; 
                float value;
                auto [ptr_ix, ec_ix] = std::from_chars(ix_view.begin(), ix_view.end(), ix);
                auto [ptr_value, ec_value] = std::from_chars(value_view.begin(), value_view.end(), value);
                if (ec_ix != std::errc{} || ec_value != std::errc{}) {
                    return std::unexpected(std::format("malformed field at line {}", line_num));
                }
                row[ix - 1] = value; // we need zero-based indices
                n_cols = std::max(n_cols, ix);
            }
        }
        x_sparse.emplace_back(row);
        line_num++;
    }
    if (!yl.size()) {
        return std::unexpected(std::format("no data"));
    }
    MatrixXf y(yl.size(), 1);
    for (auto [i, value] : std::views::enumerate(yl)){
        y(i,0) = value;
    }
    MatrixXf x = MatrixXf::Zero(std::size(y), cols.value_or(n_cols));
    for (const auto &[i, row]: std::views::enumerate(x_sparse)) {
        for (const auto &[j, value]: row) {
            x(i,j) = value;
        }
    }
    return std::make_pair(x, y);
}   


struct Params {
    float alpha;
    float epsilon;
    size_t n_iter;
    size_t burn_in;
    size_t leapfrog_steps;
    float val_accuracy;
};
 
/**
 * @brief Parses hyperparameters from a JSON configuration file.
 *
 * @param path Path to the JSON file.
 * @return std::expected<Params, std::string>: success with Params struct, or error message.
 */
std::expected<Params, std::string> hyper_params(const std::filesystem::path& path) {
    Params params;
    if (!std::filesystem::exists(path)) {
        return std::unexpected("no such file or directory: " + path.string());
    }
    std::ifstream params_file(path);
    if (!params_file.is_open()){
        return std::unexpected("unable to open the file: " + path.string());
    }
    try {
        json data = json::parse(params_file);
        params.epsilon = data["epsilon"].get<float>();
        params.alpha = data["alpha"].get<float>();
        params.val_accuracy = data["val_accuracy"].get<float>();
        params.leapfrog_steps = data["n_leaps"].get<size_t>();
        params.burn_in = data["burn_in"].get<size_t>();
        params.n_iter = data["n_iter"].get<size_t>();
        if (params.burn_in >= params.n_iter) {
            return std::unexpected("burn_in must be < n_iter");
        }
    } catch (const json::parse_error& e) {
        return std::unexpected(std::format("json syntax error: {}", e.what()));
    } catch (const json::exception& e) {
        return std::unexpected(std::format("json library error: {}", e.what()));
    }
    return params;
}


/**
 * @brief Main function: train HMC sampler on a9a dataset, validate on test set.
 *
 * Flow:
 *   1. Load training data (`a9a`)
 *   2. Parse hyperparameters (`params.json`)
 *   3. Warm up HMC chain
 *   4. Run full HMC sampling
 *   5. Compute posterior mean (after burn-in)
 *   6. Load test data (`a9a.t`)
 *   7. Predict and compute accuracy
 *   8. Check if accuracy meets threshold
 *   9. Output runtime via `std::println`
 *
 * @return EXIT_SUCCESS if all passes, otherwise EXIT_FAILURE.
 */
int main(){    
    const auto train_data = read_svm_light("a9a");
    if (train_data.has_value()) {
        const auto [x, y] = train_data.value();
        const auto params_data = hyper_params("params.json");
        if (!params_data.has_value()){
            throw(std::runtime_error(std::format("Error parsing hyper parameters, {}", params_data.error())));
        }
        const auto params = params_data.value();
        
        // Warm-up run (discard results)
        hmc(y, x, params.epsilon, params.leapfrog_steps, params.alpha, 1);
        
        // Perform full HMC sampling & measure runtime
        const auto time0 = std::chrono::steady_clock::now();
        const auto samples = hmc(y, x, params.epsilon, params.leapfrog_steps, params.alpha, params.n_iter);
        const auto time1 = std::chrono::steady_clock::now();
        const auto runtime = std::chrono::duration_cast<std::chrono::duration<double>>(time1 - time0);
        
        // Compute posterior mean (after burn-in)        
        const auto posterior_mean = samples.bottomRows(samples.rows() - params.burn_in).colwise().mean();
        
        // Load test data
        const auto test_data = read_svm_light("a9a.t", x.cols());
        if (test_data.has_value()) {
            const auto [x_test, y_test] = test_data.value();
            // # Predict on test set
            const auto pred = (sigmoid(x_test*posterior_mean.transpose()).array() > 0.5).cast<float>();
            const auto accuracy = (pred == y_test.array()).cast<float>().mean();
            // Validate accuracy against threshold
            if (accuracy < params.val_accuracy) {
                throw std::logic_error(std::format("Error: Accuracy is too low: {}", accuracy));
            }
            // Output runtime as JSON-compatible string
            std::println("{{c++-eigen: {}}}", runtime.count());
            return EXIT_SUCCESS;
        } else {
            throw(std::runtime_error(std::format("Error loading test data, {}", test_data.error())));
        }
    } else {
        throw(std::runtime_error(std::format("Error loading train data, {}", train_data.error())));
    }
    return EXIT_FAILURE;
}


