use ndarray::{Array, Array2, Axis, s};
use std::fs::File;
use std::path::Path;
use std::io::{BufRead, BufReader};
use std::time::{Instant};
extern crate ndarray;
extern crate ndarray_rand;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::random;
use ndarray_rand::rand_distr::Normal;
use std::collections::HashMap;
use std::io::{Error, ErrorKind};
use serde::Deserialize;


/// Sigmoid function: `σ(x) = 1 / (1 + exp(-x))`
/// 
/// This is applied element-wise to the input array.
/// 
/// # Arguments
/// * `x` - Input array
/// 
/// # Returns
/// A new `Array2<f32>` with the same dimensions as `x`, where each element is `1 / (1 + exp(-x))`
fn sigmoid(x:&Array2<f32>) -> Array2<f32> {
    1f32/(1f32 + (-x).exp())
}

/// Numerically stable softplus function: `softplus(x) = log(1 + exp(x))`
/// 
/// # Arguments
/// * `x` - Input array
/// 
/// # Returns
/// A new `Array2<f32>` with the same dimensions as `x`
fn softplus(x:&Array2<f32>) -> Array2<f32> {
    x.mapv(|xi| xi.max(0f32)) + (-x.abs()).exp().ln_1p()
}


/// Computes the negative log-posterior (potential energy) U(beta | y, x, alpha).
/// 
/// # Arguments
/// * `y` - Target labels as a column vector of shape `(N, 1)`
/// * `x` - Feature matrix of shape `(N, D)`
/// * `beta` - Coefficient vector (parameters) of shape `(D, 1)`
/// * `alpha` - Prior precision (inverse variance) of the Gaussian prior on `beta`
/// 
/// # Returns
/// Scalar value of the potential energy `U(β)`
fn u(y: &Array2<f32>, x: &Array2<f32>, beta: &Array2<f32>, alpha: f32) -> f32 {
    let x_beta = x.dot(beta);
    (softplus(&x_beta).sum() - y.t().dot(&x_beta) + beta.t().dot(beta)/(2f32*alpha))[[0,0]]
}


/// Computes the gradient of the potential energy ∇_β U with respect to β.
/// 
/// # Arguments
/// * `y` - Target labels `(N, 1)`
/// * `x` - Feature matrix `(N, D)`
/// * `beta` - Current parameters `(D, 1)`
/// * `alpha` - Prior precision (1/prior variance)
/// 
/// # Returns
/// Gradient vector of shape `(D, 1)`
fn grad_u(y: &Array2<f32>, x: &Array2<f32>, beta: &Array2<f32>, alpha: f32) -> Array2<f32> {
    x.t().dot(&(&sigmoid(&x.dot(beta)) - y)) + beta/alpha
}


/// Perform one HMC step using the Leapfrog integrator.
/// 
/// # Arguments
/// * `y` - Training labels `(N, 1)`
/// * `x` - Training features `(N, D)`
/// * `epsilon` - Step size for the Leapfrog integrator
/// * `leapfrog_steps` - Number of Leapfrog steps per HMC iteration
/// * `current_q` - Current parameter value (position) `(D, 1)`
/// * `alpha` - Prior precision (controls regularization strength)
/// 
/// # Returns
/// The proposed `beta` vector, or the old one if rejected
fn hmc_step(y: &Array2<f32>, x: &Array2<f32>, epsilon: f32, leapfrog_steps: usize, current_q: &Array2<f32>, alpha: f32) -> Array2<f32> {
    // Initialize momentum p and current state q
    let mut p: ndarray::Array2<f32> = Array::random(current_q.raw_dim(), Normal::new(0f32, 1f32).unwrap());
    let current_p = p.clone();
    let mut q = current_q.clone();
    
    // Half-step momentum update
    p = p - 0.5*epsilon*grad_u(&y, &x, &q, alpha);
    for i in 0..leapfrog_steps { 
        // Position update
        q = q + epsilon*&p;
        // Momentum update (except last step)
        if i < leapfrog_steps - 1 {
            p = p - epsilon*grad_u(&y, &x, &q, alpha);
        }
    }
    // Final half-step momentum update (negated for symmetry)
    p = epsilon*grad_u(&y, &x, &q, alpha)/2f32 - p;
    // Compute potential and kinetic energies at current and proposed states
    let current_u = u(&y, &x, &current_q, alpha);
    let current_k = (current_p.t().dot(&current_p))[[0,0]]/2f32;
    let proposed_u = u(&y, &x, &q, alpha);
    let proposed_k = (p.t().dot(&p))[[0,0]]/2f32;
    let rnd = random::<f32>();
    // Metropolis acceptance
    if rnd.ln() < (current_u - proposed_u + current_k - proposed_k) { 
        q   // Accept proposal
    }  else {
        current_q.clone()  // Reject: stay at current state
    }
}


/// Runs Hamiltonian Monte Carlo (HMC) sampler to generate posterior samples.
/// 
/// # Arguments
/// * `y` - Training labels `(N, 1)`
/// * `x` - Training features `(N, D)`
/// * `epsilon` - Step size for Leapfrog integrator
/// * `leapfrog_steps` - Number of Leapfrog steps per iteration
/// * `alpha` - Prior precision (controls regularization)
/// * `n_iter` - Total number of HMC iterations
/// 
/// # Returns
/// A matrix of shape `(n_iter, D)` containing all sampled `beta` vectors
fn hmc(y: &Array2<f32>, x: &Array2<f32>, epsilon: f32, leapfrog_steps: usize, alpha: f32, n_iter: usize) -> Array2<f32> {
    let mut q:Array2<f32> = Array2::zeros((x.ncols(), 1));
    let mut samples:Array2<f32> = Array2::zeros((n_iter, x.ncols()));
    for i in 0..n_iter{
        q = hmc_step(&y, &x, epsilon, leapfrog_steps, &q, alpha);
        samples.row_mut(i).assign(&q.column(0));
    }
    samples
}


/// Reads a SVM-light formatted dataset from file.
/// 
/// # Arguments
/// * `path` - Path to the SVMlight file
/// * `cols` - Optional number of features. If `None`, inferred from data.
/// 
/// # Returns
/// A tuple `(x, y)` where:
/// - `x` is a dense feature matrix of shape `(N, D)`
/// - `y` is a column vector of targets of shape `(N, 1)`
/// 
/// # Errors
/// Returns `Err` if file is malformed, contains invalid values, or is empty.
fn read_svm_light(path: &Path, cols: Option<usize>) -> Result<(Array2<f32>, Array2<f32>), Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut yl = Vec::new();
    let mut x_sparse = Vec::new();
    let mut n_cols:usize = 0;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let mut row: HashMap<usize, f32> = HashMap::new();
        for (ix, field) in line.split_whitespace().enumerate() {
            if ix == 0 {
                let yf: f32 = field.parse().map_err(|_| Error::new(ErrorKind::InvalidData, format!("Bad label at line {}", line_num)))?;
                yl.push(yf.max(0f32));
            } else {
                let (ix_str, value_str) = field.split_once(':').ok_or_else(|| Error::new(ErrorKind::InvalidData, "Bad line"))?;
                let ix:usize = ix_str.parse().map_err(|_| Error::new(ErrorKind::InvalidData, "Bad field"))?;
                let value:f32 = value_str.parse().map_err(|_| Error::new(ErrorKind::InvalidData, "Bad field"))?;
                row.insert(ix - 1,  value); // we need zero-based indices
                n_cols = n_cols.max(ix);
            }
        }
        x_sparse.push(row);
    }
    if yl.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, "no data!"));
    }
    let nrows = yl.len();
    let ncols = cols.unwrap_or(n_cols);

    let y = Array2::from_shape_vec((yl.len(), 1), yl).unwrap();
    let mut x = Array2::<f32>::zeros((nrows, ncols));
    for (i, row) in x_sparse.iter().enumerate() {
        for (&j, val) in row {
            if j < ncols {
                x[[i, j]] = *val;
            }
        }
    }    
    Ok((x, y))
}


/// Configuration structure for HMC parameters and validation criteria
/// 
/// This structure is deserialized from a JSON file `params.json`.
#[derive(Deserialize, Debug)]
struct Params {
    alpha: f32,
    epsilon: f32,
    n_iter: usize,
    burn_in: usize,
    n_leaps: usize,
    val_accuracy: f32,
}


/// Load hyperparameters from `params.json`
/// 
/// # Returns
/// `Ok(Params)` if successful, `Err` otherwise
fn get_params() -> Result<Params, Box<dyn std::error::Error>> {
    let file = File::open("params.json")?;
    let reader = BufReader::new(file);
    let params: Params = serde_json::from_reader(reader)?;
    Ok(params)
}


/// Main function: run HMC for logistic regression, evaluate on test set
/// 
/// This function:
/// 1. Loads training data
/// 2. Reads HMC hyperparameters
/// 3. Runs HMC to sample posterior distribution of β
/// 4. Computes posterior mean (after burn-in)
/// 5. Loads test data and makes predictions
/// 6. Computes accuracy and validates against threshold
/// 7. Prints execution time in a formatted JSON output
fn main() {
    match read_svm_light(Path::new("a9a"), None) { //load_svmlight_to_ndarray("a9a") {
        Ok((x, y)) => {
            let params = get_params().expect("Unable to load the hyper parameters!");
            
            // Warm-up run (discard results)
            hmc(&y, &x, params.epsilon, params.n_leaps, params.alpha, 1);
            
            // Perform full HMC sampling & measure runtime
            let t0 = Instant::now();
            let samples = hmc(&y, &x, params.epsilon, params.n_leaps, params.alpha, params.n_iter);
            let time_delta = (Instant::now() - t0).as_millis() as f32/1000.;
            
            // Compute posterior mean (after burn-in)
            let posterior_mean = samples.slice(s![params.burn_in.., ..]).mean_axis(Axis(0));
            
            // Load test data
            match read_svm_light(Path::new("a9a.t"), Some(x.ncols())) {
                Ok((x_test, y_test)) => {
                    // # Predict on test set
                    let coef =  posterior_mean.unwrap().insert_axis(Axis(1));
                    let pred = sigmoid(&x_test.dot(&coef)).map(|p| *p > 0.5f32);
                    let accuracy = pred.iter().zip(&y_test).filter(|(p, t)| **p==(**t > 0f32)).count() as f32/pred.len() as f32;             // Validate accuracy against threshold
                    if accuracy < params.val_accuracy {
                        panic!("Accuracy is too low: {}", accuracy);
                    }
                    // Output runtime as JSON-compatible string
                    println!("{{rust-ndarray: {}}}", time_delta);
                }
                Err(e) => {
                    eprintln!("Error loading test data: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Error loading train data: {}", e);
        }
    }
}


