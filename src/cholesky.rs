//! Cholesky decomposition of symmetric positive-definite matrices.

use crate::error::{LinalgError, Result};
use crate::matrix::Matrix;
use crate::sum;
use crate::vector::Vector;

/// Two mirrored elements may differ by this fraction of the largest magnitude in the
/// matrix and still count as equal when symmetry is checked.
pub const SYMMETRY_TOLERANCE: f64 = 1e-12;

/// The factorisation `A = L * L^T` with `L` lower triangular and a positive diagonal.
///
/// It exists exactly when `A` is symmetric positive definite, which makes the decomposition
/// a practical test for that property. It costs half as much as an LU decomposition and
/// needs no pivoting.
#[derive(Debug, Clone, PartialEq)]
pub struct Cholesky {
    l: Matrix,
}

impl Cholesky {
    /// Decomposes `a`.
    ///
    /// Fails with [`LinalgError::NotSquare`], [`LinalgError::Empty`],
    /// [`LinalgError::NotSymmetric`] (see [`SYMMETRY_TOLERANCE`]) or, when a pivot is zero,
    /// negative or NaN, [`LinalgError::NotPositiveDefinite`].
    pub fn new(a: &Matrix) -> Result<Cholesky> {
        a.require_square()?;
        if a.is_empty() {
            return Err(LinalgError::Empty {
                op: "Cholesky decomposition",
            });
        }
        if let Some((row, col)) = a.first_asymmetry(SYMMETRY_TOLERANCE * a.norm_max()) {
            return Err(LinalgError::NotSymmetric { row, col });
        }
        let n = a.rows();
        let mut l = Matrix::zeros(n, n);
        for j in 0..n {
            let diagonal = a[(j, j)] - sum::dot(&l.row(j)[..j], &l.row(j)[..j]);
            if !(diagonal > 0.0) {
                return Err(LinalgError::NotPositiveDefinite { pivot: j });
            }
            let pivot = diagonal.sqrt();
            l[(j, j)] = pivot;
            for i in (j + 1)..n {
                let value = a[(i, j)] - sum::dot(&l.row(i)[..j], &l.row(j)[..j]);
                l[(i, j)] = value / pivot;
            }
        }
        Ok(Cholesky { l })
    }

    /// The order of the decomposed matrix.
    pub fn order(&self) -> usize {
        self.l.rows()
    }

    /// The lower triangular factor.
    pub fn l(&self) -> &Matrix {
        &self.l
    }

    /// The determinant of the decomposed matrix: the squared product of the diagonal of `L`.
    pub fn determinant(&self) -> f64 {
        let product: f64 = self.l.diagonal().iter().product();
        product * product
    }

    /// The natural logarithm of the determinant, which stays finite when the determinant
    /// itself would overflow or underflow.
    pub fn log_determinant(&self) -> f64 {
        2.0 * sum::sum(self.l.diagonal().iter().map(|d| d.ln()))
    }

    /// Solves `A * x = b` by one forward and one backward substitution.
    pub fn solve(&self, b: &Vector) -> Result<Vector> {
        let n = self.order();
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch {
                op: "Cholesky solve",
                left: (n, n),
                right: (b.len(), 1),
            });
        }
        let mut x = b.as_slice().to_vec();
        for i in 0..n {
            let row = self.l.row(i);
            x[i] = (x[i] - sum::dot(&row[..i], &x[..i])) / row[i];
        }
        for i in (0..n).rev() {
            let mut value = x[i];
            for k in (i + 1)..n {
                value -= self.l[(k, i)] * x[k];
            }
            x[i] = value / self.l[(i, i)];
        }
        Ok(Vector::new(x))
    }

    /// The inverse of the decomposed matrix, column by column.
    pub fn inverse(&self) -> Result<Matrix> {
        let n = self.order();
        let mut inverse = Matrix::zeros(n, n);
        for j in 0..n {
            let column = self.solve(&Vector::unit(n, j))?;
            for i in 0..n {
                inverse[(i, j)] = column[i];
            }
        }
        Ok(inverse)
    }
}

impl Matrix {
    /// The Cholesky decomposition; see [`Cholesky::new`].
    pub fn cholesky(&self) -> Result<Cholesky> {
        Cholesky::new(self)
    }
}
