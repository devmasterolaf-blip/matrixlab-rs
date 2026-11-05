//! QR decomposition by Householder reflections, and least squares.

use crate::error::{LinalgError, Result};
use crate::matrix::Matrix;
use crate::sum;
use crate::vector::Vector;

/// The factorisation `A = Q * R` of an `m` x `n` matrix with `m >= n`.
///
/// `Q` has orthonormal columns and `R` is upper triangular. `Q` is kept as the list of
/// Householder reflections that produced `R`, which is both cheaper and more accurate than
/// forming it; [`Qr::q`] builds the explicit matrix when it is wanted.
#[derive(Debug, Clone, PartialEq)]
pub struct Qr {
    /// The reduced matrix; its upper triangle is `R`.
    reduced: Matrix,
    /// Reflection `k` is `I - 2 * v * v^T` with the unit vector `v` acting on rows `k..m`.
    /// An empty vector stands for the identity (the column was already zero).
    reflectors: Vec<Vec<f64>>,
}

/// The result of a least-squares solve.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquares {
    /// The vector `x` that minimises `||A * x - b||`.
    pub solution: Vector,
    /// The Euclidean norm of the residual `A * x - b` at the minimum.
    pub residual_norm: f64,
}

impl Qr {
    /// Decomposes `a`.
    ///
    /// Fails with [`LinalgError::Empty`] for a matrix without elements and with
    /// [`LinalgError::Underdetermined`] when it has fewer rows than columns.
    pub fn new(a: &Matrix) -> Result<Qr> {
        if a.is_empty() {
            return Err(LinalgError::Empty {
                op: "QR decomposition",
            });
        }
        let (m, n) = a.shape();
        if m < n {
            return Err(LinalgError::Underdetermined { rows: m, cols: n });
        }
        let mut reduced = a.clone();
        let mut reflectors = Vec::with_capacity(n);
        for k in 0..n {
            let mut v: Vec<f64> = (k..m).map(|i| reduced[(i, k)]).collect();
            let norm = sum::norm2(&v);
            if norm == 0.0 {
                reflectors.push(Vec::new());
                continue;
            }
            // Reflect onto the axis with the sign opposite to the leading element, so the
            // subtraction below adds magnitudes instead of cancelling them.
            let alpha = if v[0] >= 0.0 { -norm } else { norm };
            v[0] -= alpha;
            let v_norm = sum::norm2(&v);
            for x in v.iter_mut() {
                *x /= v_norm;
            }
            for j in (k + 1)..n {
                let mut projection = sum::CompensatedSum::new();
                for (offset, &vi) in v.iter().enumerate() {
                    projection.add(vi * reduced[(k + offset, j)]);
                }
                let twice = 2.0 * projection.value();
                for (offset, &vi) in v.iter().enumerate() {
                    reduced[(k + offset, j)] -= twice * vi;
                }
            }
            reduced[(k, k)] = alpha;
            for i in (k + 1)..m {
                reduced[(i, k)] = 0.0;
            }
            reflectors.push(v);
        }
        Ok(Qr {
            reduced,
            reflectors,
        })
    }

    /// The shape `(m, n)` of the decomposed matrix.
    pub fn shape(&self) -> (usize, usize) {
        self.reduced.shape()
    }

    /// The upper triangular factor, of shape `n` x `n`.
    pub fn r(&self) -> Matrix {
        let n = self.reduced.cols();
        Matrix::from_fn(n, n, |i, j| if i <= j { self.reduced[(i, j)] } else { 0.0 })
    }

    /// The factor with orthonormal columns, of shape `m` x `n` (the "thin" `Q`).
    pub fn q(&self) -> Matrix {
        let (m, n) = self.shape();
        let mut q = Matrix::zeros(m, n);
        for j in 0..n {
            let mut column = vec![0.0; m];
            column[j] = 1.0;
            for k in (0..n).rev() {
                reflect(&self.reflectors[k], &mut column[k..]);
            }
            for (i, &value) in column.iter().enumerate() {
                q[(i, j)] = value;
            }
        }
        q
    }

    /// The product `Q^T * b` with the full `m` x `m` orthogonal matrix, of length `m`.
    pub fn q_transpose_mul(&self, b: &Vector) -> Result<Vector> {
        let (m, n) = self.shape();
        if b.len() != m {
            return Err(LinalgError::DimensionMismatch {
                op: "QR solve",
                left: (m, n),
                right: (b.len(), 1),
            });
        }
        let mut y = b.as_slice().to_vec();
        for (k, v) in self.reflectors.iter().enumerate() {
            reflect(v, &mut y[k..]);
        }
        Ok(Vector::new(y))
    }

    /// The first column whose diagonal element of `R` is negligible, if any.
    ///
    /// "Negligible" means at most `max(m, n) * f64::EPSILON` times the largest diagonal
    /// magnitude. Such a column is, to working precision, a combination of the columns
    /// before it.
    pub fn deficient_column(&self) -> Option<usize> {
        let (m, n) = self.shape();
        let diagonal = self.reduced.diagonal();
        let largest = diagonal.iter().fold(0.0_f64, |acc, d| acc.max(d.abs()));
        let threshold = m.max(n) as f64 * f64::EPSILON * largest;
        diagonal.iter().position(|d| !(d.abs() > threshold))
    }

    /// Solves the least-squares problem `min ||A * x - b||`.
    ///
    /// For a square matrix this is an ordinary linear solve. Fails with
    /// [`LinalgError::RankDeficient`] when the columns of `A` are linearly dependent, since
    /// the minimiser is then not unique.
    pub fn solve_least_squares(&self, b: &Vector) -> Result<LeastSquares> {
        let y = self.q_transpose_mul(b)?;
        if let Some(column) = self.deficient_column() {
            return Err(LinalgError::RankDeficient { column });
        }
        let n = self.reduced.cols();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let row = self.reduced.row(i);
            let mut value = y[i];
            for j in (i + 1)..n {
                value -= row[j] * x[j];
            }
            x[i] = value / row[i];
        }
        Ok(LeastSquares {
            solution: Vector::new(x),
            residual_norm: sum::norm2(&y.as_slice()[n..]),
        })
    }
}

/// Applies the reflection `I - 2 * v * v^T` to `x` in place; an empty `v` is the identity.
fn reflect(v: &[f64], x: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let twice = 2.0 * sum::dot(v, x);
    for (xi, &vi) in x.iter_mut().zip(v) {
        *xi -= twice * vi;
    }
}

/// Solves `min ||a * x - b||` by Householder QR; see [`Qr::solve_least_squares`].
pub fn lstsq(a: &Matrix, b: &Vector) -> Result<LeastSquares> {
    Qr::new(a)?.solve_least_squares(b)
}

impl Matrix {
    /// The QR decomposition; see [`Qr::new`].
    pub fn qr(&self) -> Result<Qr> {
        Qr::new(self)
    }
}
