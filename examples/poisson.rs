//! Solves the one-dimensional Poisson problem `-u'' = 1` on (0, 1) with `u(0) = u(1) = 0`,
//! discretised with second differences, and compares the conjugate-gradient solution of the
//! sparse system with the exact one, `u(x) = x (1 - x) / 2`. Second differences are exact
//! for a parabola, so the error that is printed is rounding error only.
//!
//! Run it with `cargo run --example poisson`, optionally followed by the number of interior
//! grid points.

use matrixlab::iterative::{conjugate_gradient, IterativeOptions};
use matrixlab::sparse::CooMatrix;
use matrixlab::Vector;

fn main() {
    let n: usize = match std::env::args().nth(1) {
        None => 50,
        Some(text) => match text.parse() {
            Ok(n) if n > 0 => n,
            _ => {
                eprintln!("the number of grid points must be a positive whole number");
                std::process::exit(2);
            }
        },
    };
    let h = 1.0 / (n + 1) as f64;

    let mut assembly = CooMatrix::new(n, n);
    for i in 0..n {
        assembly.push(i, i, 2.0).expect("the index is inside the matrix");
        if i + 1 < n {
            assembly.push(i, i + 1, -1.0).expect("the index is inside the matrix");
            assembly.push(i + 1, i, -1.0).expect("the index is inside the matrix");
        }
    }
    let a = assembly.to_csr();
    let b = Vector::filled(n, h * h);

    match conjugate_gradient(&a, &b, &IterativeOptions::default()) {
        Ok(result) => {
            let worst = result
                .solution
                .iter()
                .enumerate()
                .map(|(i, u)| {
                    let x = (i + 1) as f64 * h;
                    (u - x * (1.0 - x) / 2.0).abs()
                })
                .fold(0.0, f64::max);
            println!("grid points:       {}", n);
            println!("stored entries:    {}", a.nnz());
            println!("iterations:        {}", result.iterations);
            println!("residual norm:     {:.3e}", result.residual_norm);
            println!("largest error:     {:.3e}", worst);
        }
        Err(error) => {
            eprintln!("the solver failed: {}", error);
            std::process::exit(1);
        }
    }
}
