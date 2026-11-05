//! Iterative solvers for sparse linear systems.
//!
//! All three solvers share one contract. They start from the initial guess (zero unless
//! given), stop as soon as the residual satisfies `||b - A * x|| <= tolerance * ||b||` in
//! the Euclidean norm, and return [`LinalgError::NoConvergence`] with the last residual when
//! the iteration limit is reached or the iterates stop being finite. A zero right-hand side
//! gives the zero solution without iterating.

use crate::cholesky::SYMMETRY_TOLERANCE;
use crate::error::{LinalgError, Result};
use crate::sparse::CsrMatrix;
use crate::vector::Vector;

/// Stopping rule and starting point of an iterative solve.
#[derive(Debug, Clone, PartialEq)]
pub struct IterativeOptions {
    /// The largest number of iterations (sweeps for the stationary methods).
    pub max_iterations: usize,
    /// The relative residual `||b - A * x|| / ||b||` at which the solve stops.
    pub tolerance: f64,
    /// The starting vector; `None` means zero.
    pub initial_guess: Option<Vector>,
}

impl Default for IterativeOptions {
    fn default() -> IterativeOptions {
        IterativeOptions {
            max_iterations: 1000,
            tolerance: 1e-10,
            initial_guess: None,
        }
    }
}

/// The outcome of a converged iterative solve.
#[derive(Debug, Clone, PartialEq)]
pub struct IterativeSolution {
    /// The approximate solution.
    pub solution: Vector,
    /// The number of iterations that were performed.
    pub iterations: usize,
    /// The Euclidean norm of the true residual `b - A * x` of the returned solution.
    pub residual_norm: f64,
}

/// What every solver needs before its first iteration.
struct Setup {
    x: Vector,
    /// The residual norm at which the solve stops.
    target: f64,
}

/// Validates the inputs and prepares the starting vector.
fn setup(
    a: &CsrMatrix,
    b: &Vector,
    options: &IterativeOptions,
    method: &'static str,
) -> Result<Setup> {
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols });
    }
    if rows == 0 {
        return Err(LinalgError::Empty { op: method });
    }
    if b.len() != rows {
        return Err(LinalgError::DimensionMismatch {
            op: method,
            left: a.shape(),
            right: (b.len(), 1),
        });
    }
    if !(options.tolerance >= 0.0) {
        return Err(LinalgError::InvalidArgument {
            what: "the convergence tolerance must be zero or positive",
        });
    }
    let mut x = match &options.initial_guess {
        Some(guess) if guess.len() != rows => {
            return Err(LinalgError::DimensionMismatch {
                op: "initial guess",
                left: a.shape(),
                right: (guess.len(), 1),
            });
        }
        Some(guess) => guess.clone(),
        None => Vector::zeros(rows),
    };
    let b_norm = b.norm2();
    if b_norm == 0.0 {
        // The solution of a homogeneous system is zero whatever the starting vector, and a
        // relative residual cannot be measured against a zero right-hand side.
        x = Vector::zeros(rows);
    }
    Ok(Setup {
        x,
        target: options.tolerance * b_norm,
    })
}

/// The residual `b - A * x`.
fn residual(a: &CsrMatrix, b: &Vector, x: &Vector) -> Result<Vector> {
    b.try_sub(&a.mul_vec(x)?)
}

/// The diagonal of `a`, or the first row whose diagonal element is zero or missing.
fn nonzero_diagonal(a: &CsrMatrix) -> Result<Vec<f64>> {
    let diagonal = a.diagonal();
    match diagonal.iter().position(|&d| d == 0.0) {
        Some(row) => Err(LinalgError::ZeroDiagonal { row }),
        None => Ok(diagonal),
    }
}

/// Runs a stationary iteration: calls `sweep` to update `x` in place until the residual
/// meets the target.
fn stationary<F>(
    a: &CsrMatrix,
    b: &Vector,
    options: &IterativeOptions,
    method: &'static str,
    mut sweep: F,
) -> Result<IterativeSolution>
where
    F: FnMut(&mut Vector, &[f64]),
{
    let Setup { mut x, target } = setup(a, b, options, method)?;
    let diagonal = nonzero_diagonal(a)?;
    let mut residual_norm = residual(a, b, &x)?.norm2();
    let mut iterations = 0;
    while !(residual_norm <= target) {
        if iterations == options.max_iterations || !residual_norm.is_finite() {
            return Err(LinalgError::NoConvergence {
                method,
                iterations,
                residual: residual_norm,
            });
        }
        sweep(&mut x, &diagonal);
        iterations += 1;
        residual_norm = residual(a, b, &x)?.norm2();
    }
    Ok(IterativeSolution {
        solution: x,
        iterations,
        residual_norm,
    })
}

/// The sum of `a[i][j] * x[j]` over the stored entries of row `i` with `j != i`.
fn off_diagonal_dot(a: &CsrMatrix, i: usize, x: &[f64]) -> f64 {
    let (columns, values) = a.row(i);
    let mut total = 0.0;
    for (&j, &value) in columns.iter().zip(values) {
        if j != i {
            total += value * x[j];
        }
    }
    total
}

/// Solves `A * x = b` by Jacobi iteration.
///
/// Every component is updated from the previous iterate:
/// `x[i] = (b[i] - sum_{j != i} a[i][j] * x[j]) / a[i][i]`. The method converges for
/// strictly diagonally dominant matrices and may diverge otherwise.
///
/// Fails with [`LinalgError::ZeroDiagonal`] when a diagonal element is zero or not stored.
pub fn jacobi(a: &CsrMatrix, b: &Vector, options: &IterativeOptions) -> Result<IterativeSolution> {
    stationary(a, b, options, "Jacobi iteration", |x, diagonal| {
        let previous = x.as_slice().to_vec();
        for (i, xi) in x.as_mut_slice().iter_mut().enumerate() {
            *xi = (b[i] - off_diagonal_dot(a, i, &previous)) / diagonal[i];
        }
    })
}

/// Solves `A * x = b` by Gauss-Seidel iteration.
///
/// Like Jacobi, but each component uses the components already updated in the same sweep.
/// The method converges for strictly diagonally dominant matrices and for symmetric
/// positive-definite ones, usually in about half the sweeps Jacobi needs.
pub fn gauss_seidel(
    a: &CsrMatrix,
    b: &Vector,
    options: &IterativeOptions,
) -> Result<IterativeSolution> {
    stationary(a, b, options, "Gauss-Seidel iteration", |x, diagonal| {
        for i in 0..diagonal.len() {
            let updated = (b[i] - off_diagonal_dot(a, i, x.as_slice())) / diagonal[i];
            x[i] = updated;
        }
    })
}

/// Solves `A * x = b` for a symmetric positive-definite `A` by the conjugate gradient
/// method.
///
/// Fails with [`LinalgError::NotSymmetric`] when `A` is not symmetric and with
/// [`LinalgError::NotPositiveDefinite`] when a search direction with non-positive curvature
/// shows that `A` is not positive definite; `pivot` is then the iteration at which it was
/// found.
pub fn conjugate_gradient(
    a: &CsrMatrix,
    b: &Vector,
    options: &IterativeOptions,
) -> Result<IterativeSolution> {
    let method = "conjugate gradient";
    let Setup { mut x, target } = setup(a, b, options, method)?;
    let largest = a.values().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if let Some((row, col)) = a.first_asymmetry(SYMMETRY_TOLERANCE * largest) {
        return Err(LinalgError::NotSymmetric { row, col });
    }
    let mut r = residual(a, b, &x)?;
    let mut residual_norm = r.norm2();
    let mut direction = r.clone();
    let mut rho = r.dot(&r)?;
    let mut iterations = 0;
    while !(residual_norm <= target) {
        if iterations == options.max_iterations || !residual_norm.is_finite() {
            return Err(LinalgError::NoConvergence {
                method,
                iterations,
                residual: residual_norm,
            });
        }
        let a_direction = a.mul_vec(&direction)?;
        let curvature = direction.dot(&a_direction)?;
        if !(curvature > 0.0) {
            return Err(LinalgError::NotPositiveDefinite { pivot: iterations });
        }
        let step = rho / curvature;
        x.axpy(step, &direction)?;
        r.axpy(-step, &a_direction)?;
        iterations += 1;

        let rho_next = r.dot(&r)?;
        if rho_next.sqrt() <= target {
            // The recurrence says the solve is done; confirm with the true residual, which
            // can differ from the recurrence after many steps.
            r = residual(a, b, &x)?;
            residual_norm = r.norm2();
            rho = r.dot(&r)?;
            direction = r.clone();
            continue;
        }
        residual_norm = rho_next.sqrt();
        direction = &r + &direction.scale(rho_next / rho);
        rho = rho_next;
    }
    Ok(IterativeSolution {
        solution: x,
        iterations,
        residual_norm,
    })
}
