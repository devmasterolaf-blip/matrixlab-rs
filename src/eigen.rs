//! Eigenvalues: the Jacobi method for symmetric matrices and power iteration.

use crate::cholesky::SYMMETRY_TOLERANCE;
use crate::error::{LinalgError, Result};
use crate::matrix::Matrix;
use crate::vector::Vector;

/// Limits for the Jacobi eigenvalue iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobiOptions {
    /// The largest number of sweeps over all off-diagonal elements.
    pub max_sweeps: usize,
    /// The iteration stops when the Frobenius norm of the off-diagonal part is at most this
    /// fraction of the Frobenius norm of the matrix.
    pub tolerance: f64,
}

impl Default for JacobiOptions {
    fn default() -> JacobiOptions {
        JacobiOptions {
            max_sweeps: 64,
            tolerance: 1e-15,
        }
    }
}

/// The eigenvalues and eigenvectors of a symmetric matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricEigen {
    /// The eigenvalues in ascending order.
    pub values: Vec<f64>,
    /// The eigenvectors as columns, in the order of `values`. They are orthonormal, and each
    /// one is signed so that its largest component is positive.
    pub vectors: Matrix,
    /// The number of sweeps that were needed.
    pub sweeps: usize,
}

/// Computes all eigenvalues and eigenvectors of a symmetric matrix with default limits.
pub fn symmetric_eigen(a: &Matrix) -> Result<SymmetricEigen> {
    symmetric_eigen_with(a, &JacobiOptions::default())
}

/// Computes all eigenvalues and eigenvectors of a symmetric matrix by cyclic Jacobi
/// rotations.
///
/// Each rotation zeroes one off-diagonal pair; a sweep visits every pair once. The method
/// is slower than tridiagonal QR for large matrices but simple, and it computes small
/// eigenvalues to high relative accuracy.
///
/// Fails with [`LinalgError::NotSquare`], [`LinalgError::Empty`],
/// [`LinalgError::NotSymmetric`], [`LinalgError::InvalidArgument`] for a negative or NaN
/// tolerance, and [`LinalgError::NoConvergence`] when `max_sweeps` is exhausted.
pub fn symmetric_eigen_with(a: &Matrix, options: &JacobiOptions) -> Result<SymmetricEigen> {
    a.require_square()?;
    if a.is_empty() {
        return Err(LinalgError::Empty {
            op: "eigenvalue decomposition",
        });
    }
    if !(options.tolerance >= 0.0) {
        return Err(LinalgError::InvalidArgument {
            what: "the eigenvalue tolerance must be zero or positive",
        });
    }
    if let Some((row, col)) = a.first_asymmetry(SYMMETRY_TOLERANCE * a.norm_max()) {
        return Err(LinalgError::NotSymmetric { row, col });
    }
    let n = a.rows();
    let threshold = options.tolerance * a.norm_frobenius();
    let mut work = a.clone();
    let mut vectors = Matrix::identity(n);
    let mut sweeps = 0;
    loop {
        let off = off_diagonal_norm(&work);
        if off <= threshold {
            break;
        }
        if sweeps == options.max_sweeps {
            return Err(LinalgError::NoConvergence {
                method: "Jacobi eigenvalue iteration",
                iterations: sweeps,
                residual: off,
            });
        }
        for p in 0..n {
            for q in (p + 1)..n {
                rotate(&mut work, &mut vectors, p, q);
            }
        }
        sweeps += 1;
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| work[(i, i)].total_cmp(&work[(j, j)]));
    let values = order.iter().map(|&i| work[(i, i)]).collect();
    let mut sorted = Matrix::zeros(n, n);
    for (target, &source) in order.iter().enumerate() {
        let column = vectors.column(source);
        let largest = column
            .iter()
            .fold(0.0_f64, |acc, &x| if x.abs() > acc.abs() { x } else { acc });
        let sign = if largest < 0.0 { -1.0 } else { 1.0 };
        for (i, &x) in column.iter().enumerate() {
            sorted[(i, target)] = sign * x;
        }
    }
    Ok(SymmetricEigen {
        values,
        vectors: sorted,
        sweeps,
    })
}

/// The Frobenius norm of the off-diagonal part of a symmetric matrix.
fn off_diagonal_norm(a: &Matrix) -> f64 {
    let n = a.rows();
    let mut upper = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        upper.extend_from_slice(&a.row(i)[i + 1..]);
    }
    std::f64::consts::SQRT_2 * crate::sum::norm2(&upper)
}

/// Applies the Jacobi rotation in the `(p, q)` plane that zeroes `a[p][q]`, and accumulates
/// it into `vectors`.
fn rotate(a: &mut Matrix, vectors: &mut Matrix, p: usize, q: usize) {
    let apq = a[(p, q)];
    if apq == 0.0 {
        return;
    }
    let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
    // The smaller root of t^2 + 2 * theta * t - 1 = 0, i.e. a rotation of at most 45 degrees.
    let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
    let c = 1.0 / t.hypot(1.0);
    let s = t * c;
    let n = a.rows();
    for k in 0..n {
        let (akp, akq) = (a[(k, p)], a[(k, q)]);
        a[(k, p)] = c * akp - s * akq;
        a[(k, q)] = s * akp + c * akq;
    }
    for k in 0..n {
        let (apk, aqk) = (a[(p, k)], a[(q, k)]);
        a[(p, k)] = c * apk - s * aqk;
        a[(q, k)] = s * apk + c * aqk;
    }
    // The pair is zero by construction; storing it removes the rounding residue.
    a[(p, q)] = 0.0;
    a[(q, p)] = 0.0;
    for k in 0..n {
        let (vkp, vkq) = (vectors[(k, p)], vectors[(k, q)]);
        vectors[(k, p)] = c * vkp - s * vkq;
        vectors[(k, q)] = s * vkp + c * vkq;
    }
}

/// Limits and the starting point for [`power_iteration`].
#[derive(Debug, Clone, PartialEq)]
pub struct PowerOptions {
    /// The largest number of multiplications by the matrix.
    pub max_iterations: usize,
    /// The iteration stops when `||A * x - lambda * x|| <= tolerance * ||A * x||`.
    pub tolerance: f64,
    /// The starting vector. `None` selects a fixed pseudo-random vector, which is the same
    /// on every run.
    pub start: Option<Vector>,
}

impl Default for PowerOptions {
    fn default() -> PowerOptions {
        PowerOptions {
            max_iterations: 1000,
            tolerance: 1e-10,
            start: None,
        }
    }
}

/// One eigenvalue with its eigenvector.
#[derive(Debug, Clone, PartialEq)]
pub struct EigenPair {
    /// The eigenvalue estimate (a Rayleigh quotient).
    pub value: f64,
    /// The eigenvector estimate, of unit Euclidean length.
    pub vector: Vector,
    /// The number of multiplications by the matrix that were used.
    pub iterations: usize,
}

/// Finds the eigenvalue of largest magnitude of a square matrix by power iteration.
///
/// The matrix does not have to be symmetric, but the dominant eigenvalue must be real and
/// strictly larger in magnitude than the others; the closer the second one is, the slower
/// the convergence. When two eigenvalues share the largest magnitude (for example `2` and
/// `-2`) the iteration does not settle and the result is [`LinalgError::NoConvergence`].
pub fn power_iteration(a: &Matrix, options: &PowerOptions) -> Result<EigenPair> {
    a.require_square()?;
    if a.is_empty() {
        return Err(LinalgError::Empty {
            op: "power iteration",
        });
    }
    if !(options.tolerance >= 0.0) {
        return Err(LinalgError::InvalidArgument {
            what: "the power iteration tolerance must be zero or positive",
        });
    }
    let n = a.rows();
    let start = match &options.start {
        Some(start) if start.len() != n => {
            return Err(LinalgError::DimensionMismatch {
                op: "power iteration start vector",
                left: a.shape(),
                right: (start.len(), 1),
            });
        }
        Some(start) => start.clone(),
        None => default_start(n),
    };
    let mut x = start.normalized()?;
    let mut residual = f64::INFINITY;
    for iteration in 1..=options.max_iterations {
        let y = a.try_mul_vec(&x)?;
        let value = x.dot(&y)?;
        let y_norm = y.norm2();
        let mut difference = y.clone();
        difference.axpy(-value, &x)?;
        residual = difference.norm2();
        if residual <= options.tolerance * y_norm {
            return Ok(EigenPair {
                value,
                vector: x,
                iterations: iteration,
            });
        }
        x = y.scale(1.0 / y_norm);
    }
    Err(LinalgError::NoConvergence {
        method: "power iteration",
        iterations: options.max_iterations,
        residual,
    })
}

/// A fixed vector with components in `(0, 1]` from a small linear congruential generator.
///
/// A structured start such as all ones is orthogonal to the dominant eigenvector of many
/// structured matrices; an irregular one almost never is.
fn default_start(n: usize) -> Vector {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) + 1) as f64 / (1u64 << 53) as f64
        })
        .collect()
}
