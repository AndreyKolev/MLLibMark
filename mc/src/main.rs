extern crate serde_json;
extern crate serde;
extern crate rand;
extern crate rand_distr;
use std::time::{Instant};
use std::fs;
use rand::prelude::*;
extern crate rayon;
use rayon::prelude::*;
use std::env;
use std::process;
use std::thread;
extern crate ndarray;
extern crate ndarray_rand;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray::{s};


/// Single-thread or parallel Monte Carlo simulation for a down-and-out barrier option
/// under geometric Brownian motion.
///
/// This function estimates the price of a down-and-out barrier option by simulating
/// `n` stock price paths using geometric Brownian motion. At each time step, it checks
/// whether the underlying asset price falls to or below the barrier level `b`. If the
/// barrier is hit at any point during the path, the option is knocked out (payoff = 0).
/// If the barrier is never hit, the payoff is `max(S_T - K, 0)`, where `S_T` is the
/// final stock price. The result is the expected discounted payoff.
///
/// # Arguments
///
/// * `parallel` - If `true`, uses parallel execution via Rayon; otherwise, runs sequentially.
/// * `s0` - Initial stock price
/// * `k` - Strike price
/// * `b` - Barrier level (down-and-out)
/// * `tau` - Time to maturity (in years)
/// * `r` - Risk-free interest rate
/// * `q` - Dividend yield
/// * `v` - Volatility (annualized)
/// * `m` - Number of time steps in each path
/// * `n` - Number of Monte Carlo paths to simulate
///
/// # Returns
///
/// The expected discounted payoff (option price).
///
fn barrier(parallel:bool, s0: f32, k:f32, b:f32, tau:f32, r:f32, q:f32, v:f32,  m:usize, n:usize) -> f32 {
    let dt = tau/m as f32;
    let drift = (r - q - v*v/2.)*dt;
    let scale = v*dt.sqrt();
    let log_b = b.ln();
    let log_s0 = s0.ln();
    let payoff = || {
    	let mut rng = rand::rng();
		let normal = rand_distr::Normal::new(drift, scale).unwrap();
		let mut s = log_s0;
        let mut min_price = s;
        for _ in 0..m {
            s += rng.sample(normal);  // generate log-price increment
            min_price = min_price.min(s);  // track minimum log price observed
            if min_price <= log_b {  // check if barrier is hit
            	return 0f32;  // knock-out
            }
        }
        (s.exp() - k).max(0f32)  // payoff if barrier not hit
	};
	
	if parallel { // Use Rayon
		(-r*tau).exp()*(0..n).into_par_iter().map(|_| payoff()).sum::<f32>()/n as f32
	} else {
		(-r*tau).exp()*(0..n).map(|_| payoff()).sum::<f32>()/(n as f32)
	}
}


/// Monte Carlo simulation for a down-and-out barrier option using vectorized computation.
///
/// This function computes the price of a down-and-out barrier option via Monte Carlo
/// simulation with vectorization using the `ndarray` crate. 
///
/// # Arguments
///
/// * `s0` - Initial stock price.
/// * `k` - Strike price of the option.
/// * `b` - Barrier level (must be less than `s0` to allow knock-out).
/// * `tau` - Time to maturity (in years).
/// * `r` - Risk-free interest rate (annualized).
/// * `q` - Dividend yield (annualized).
/// * `v` - Volatility (annualized).
/// * `m` - Number of time steps per path (higher values improve accuracy).
/// * `n` - Number of Monte Carlo paths to simulate.
///
/// # Returns
///
/// The estimated option price as the average discounted payoff across all simulated paths.
///
fn barrier_nd(s0: f32, k:f32, b:f32, tau:f32, r:f32, q:f32, v:f32,  m:usize, n:usize) -> f32 {
    let dt = tau/(m as f32);
    let drift = (r - q - v*v/2.)*dt;
    let scale = v*dt.sqrt();
    // Simulate log-price increments: m steps, n paths
    let mut s:ndarray::Array2<f32> = ndarray::Array::random((m, n), Normal::new(drift, scale).unwrap());
    // Accumulate to get log-price paths
    s.accumulate_axis_inplace(ndarray::Axis(0), |&prev, curr| *curr += prev);
    s += s0.ln();
    // Compute survival mask
    let l = s.fold_axis(ndarray::Axis(0), f32::MAX, |x, y| x.min(*y)).map(|x| f32::from(*x > b.ln()));
	// Compute final stock prices and payoffs
	let payoffs = l*(s.slice(s![-1, ..]).mapv(|x| (x.exp() - k).max(0.)));
	// Discount the average payoff
	(-r*tau).exp()*payoffs.mean().unwrap()
}


/// Helper function to compute the sum of payoffs for a batch of Monte Carlo paths.
///
/// # Arguments
/// * `drift` - Drift term of log-price: `(r - q - v²/2) * dt`
/// * `scale` - Volatility scaling: `v * sqrt(dt)`
/// * `log_s0` - Log of initial stock price
/// * `log_b` - Log of barrier level
/// * `k` - Strike price
/// * `m` - Number of time steps per path
/// * `n` - Number of paths to simulate
///
/// # Returns
/// Sum of payoffs across all simulated paths (not discounted).
fn payoff(drift:f32, scale:f32, log_s0:f32, log_b:f32,  k:f32, m:usize, n:usize) -> f32 {
    let mut rng = rand::rng();
	let normal = rand_distr::Normal::new(drift, scale).unwrap();
	let mut payoffs = 0f32;
	for _ in 0..n{
        let mut log_price = log_s0;
        let mut barrier_hit = false;
        for _ in 0..m {
            log_price += rng.sample(normal);  // generate log-price increment
            if log_price <= log_b {  // check if barrier is hit
            	barrier_hit = true;  // knock-out
            	break
            }
        }
        if !barrier_hit {
            payoffs += (log_price.exp() - k).max(0f32);  // payoff if barrier not hit
        }
	};
    payoffs
}


/// Parallel Monte Carlo pricing of a down-and-out barrier option using thread pooling.
///
/// Splits the simulation into multiple threads, each running a portion of the paths.
/// Each thread calls `payoff` to compute the sum of non-knocked-out payoffs. The total
/// is discounted at the risk-free rate and averaged to produce the final option price.
///
/// # Arguments
/// * `s0` - Initial stock price
/// * `k` - Strike price
/// * `b` - Barrier level (down-and-out)
/// * `tau` - Time to maturity (in years)
/// * `r` - Risk-free interest rate
/// * `q` - Dividend yield
/// * `v` - Volatility (annualized)
/// * `m` - Number of time steps per path
/// * `n` - Total number of Monte Carlo paths
///
/// # Returns
/// The estimated option price: discounted average payoff.
///
fn barrier_threads(s0: f32, k:f32, b:f32, tau:f32, r:f32, q:f32, v:f32,  m:usize, n:usize) -> f32 {
    let dt = tau/m as f32;
    let drift = (r - q - v*v/2.)*dt;
    let scale = v*dt.sqrt();
    let log_b = b.ln();
    let log_s0 = s0.ln();
    let mut threads = vec![];
    let n_threads = std::thread::available_parallelism().unwrap_or(std::num::NonZero::new(1).unwrap()).get();
    let batch = n/n_threads;
    let mut payoffs = 0f32;
    for i in 0..n_threads {  // launch worker threads distributing the load
        let ni = batch + (i < n%n_threads) as usize;
        threads.push(thread::spawn(move || {payoff(drift, scale, log_s0, log_b, k, m, ni)}));
    }
    for t in threads {
        payoffs += t.join().unwrap();
    }
    // Discount the average payoff
    (-r*tau).exp()*payoffs/(n as f32)
}

fn main() {
	let args: Vec<String> = env::args().skip(1).collect();
	let mut mode = "std";
	
    if let Some(pos) = args.iter().position(|x| x == "-mode") {
        if pos + 1 < args.len() {
            mode = &args[pos + 1];
            if !matches!(mode, "std"|"rayon"|"threads"|"ndarray") {
            	eprintln!("Error, wrong mode: {}", mode);
            	process::exit(1);
            }
        } else {
            eprintln!("Error! please provide mode parameter!");
            process::exit(1);
        }
    };
    
    let datastr = fs::read_to_string("data.json").expect("Unable to read file");
    let res: serde_json::Value = serde_json::from_str(&datastr).expect("Unable to parse");
    
    let s0 = res["price"].as_f64().unwrap() as f32; //s0
    let rate = res["rate"].as_f64().unwrap() as f32;  //r
    let k = res["strike"].as_f64().unwrap() as f32; //k
    let vol = res["vol"].as_f64().unwrap() as f32;    //v
    let tau = res["tau"].as_f64().unwrap() as f32;
    let time_steps = res["time_steps"].as_u64().unwrap() as usize;  // m
    let tol = res["tol"].as_f64().unwrap() as f32;
    let b = res["barrier"].as_f64().unwrap() as f32;
    let dy = res["dy"].as_f64().unwrap() as f32;   //div.yield
    let n_rep = res["n_rep"].as_u64().unwrap() as usize;  // n
    let val = res["val"].as_f64().unwrap() as f32;  // expected value
    let t0 = Instant::now();
    
    let barrier_price = match mode {
        "ndarray" => barrier_nd(s0, k, b, tau, rate, dy, vol, time_steps, n_rep),
        "threads" => barrier_threads(s0, k, b, tau, rate, dy, vol, time_steps, n_rep),
        "rayon" => barrier(true, s0, k, b, tau, rate, dy, vol, time_steps, n_rep),
        "std" => barrier(false, s0, k, b, tau, rate, dy, vol, time_steps, n_rep),
        _ => panic!("Unexpected mode: {}", mode)
    };
    
    let time_delta = (Instant::now() - t0).as_millis() as f32/1000.;
    assert!((barrier_price - val).abs() < tol, "The estimated price of the option differs significantly from the expected value!");
    println!("{{\"rust-{}\": {}}}", mode, time_delta)
}
