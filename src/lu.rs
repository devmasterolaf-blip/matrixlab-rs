//! LU decomposition with partial pivoting.

use crate::error::{LinalgError, Result};
use crate::matrix::Matrix;
use crate::vector::Vector;

/// The factorisation `P * A = L * U` of a square matrix.
///
/// `L` is unit lower triangular, `U` is upper triangular and `P` is a row permutation chosen
/// so that every pivot is the largest element (in magnitude) of its column. The factors are
/// stored packed in a single matrix.
///
/// Decomposing a singular matrix is not an error: the determinant of such a matrix is still
/// wanted. The decomposition records the first pivot that fell below the tolerance, and
/// [`Lu::solve`] and [`Lu::inverse`] refuse to continue with [`LinalgError::Singular`].
#[derive(Debug, Clone, PartialEq)]
pub struct Lu {
    packed: Matrix,
    permutation: Vec<usize>,
    sign: f64,
    singular_pivot: Option<usize>,
}

impl Lu {
    /// Decomposes `a` with the default singularity tolerance, `n * f64::EPSILON`.
    pub fn new(a: &Matrix) -> Result<Lu> {
        Lu::with_tolerance(a, a.rows() as f64 * f64::EPSILON)
    }

    /// Decomposes `a`, treating a pivot as zero when its magnitude is at most
    /// `tolerance` times the largest magnitude in `a`.
    ///
    /// Fails with [`LinalgError::NotSquare`], [`LinalgError::Empty`] for a 0 x 0 matrix and
    /// [`LinalgError::InvalidArgument`] for a negative or NaN tolerance.
    pub fn with_tolerance(a: &Matrix, tolerance: f64) -> Result<Lu> {
        a.require_square()?;
        if a.is_empty() {
            return Err(LinalgError::Empty {
                op: "LU decomposition",
            });
        }
        if !(tolerance >= 0.0) {
            return Err(LinalgError::InvalidArgument {
                what: "the singularity tolerance must be zero or positive",
            });
        }
        let n = a.rows();
        let threshold = tolerance * a.norm_max();
        let mut packed = a.clone();
        let mut permutation: Vec<usize> = (0..n).collect();
        let mut sign = 1.0;
        let mut singular_pivot = None;

        for k in 0..n {
            let mut pivot_row = k;
            let mut pivot_abs = packed[(k, k)].abs();
            for i in (k + 1)..n {
                let candidate = packed[(i, k)].abs();
                if candidate > pivot_abs {
                    pivot_row = i;
                    pivot_abs = candidate;
                }
            }
            // The negated comparison also catches a NaN pivot.
            if !(pivot_abs > threshold) {
                singular_pivot.get_or_insert(k);
                if !(pivot_abs > 0.0) {
                    // Nothing to eliminate below an exactly zero column.
                    continue;
                }
            }
            if pivot_row != k {
                packed.swap_rows(k, pivot_row);
                permutation.swap(k, pivot_row);
                sign = -sign;
            }
            let pivot = packed[(k, k)];
            for i in (k + 1)..n {
                let factor = packed[(i, k)] / pivot;
                packed[(i, k)] = factor;
                if factor != 0.0 {
                    for j in (k + 1)..n {
                        let upper = packed[(k, j)];
                        packed[(i, j)] -= factor * upper;
                    }
                }
            }
        }
        Ok(Lu {
            packed,
            permutation,
            sign,
            singular_pivot,
        })
    }

    /// The order of the decomposed matrix.
    pub fn order(&self) -> usize {
        self.packed.rows()
    }

    /// True when a pivot fell below the tolerance.
    pub fn is_singular(&self) -> bool {
        self.singular_pivot.is_some()
    }

    /// The index of the first pivot that fell below the tolerance, if any.
    pub fn singular_pivot(&self) -> Option<usize> {
        self.singular_pivot
    }

    /// The row permutation: row `i` of `P * A` is row `permutation()[i]` of `A`.
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// The permutation as a matrix `P`.
    pub fn permutation_matrix(&self) -> Matrix {
        let n = self.order();
        let mut p = Matrix::zeros(n, n);
        for (i, &source) in self.permutation.iter().enumerate() {
            p[(i, source)] = 1.0;
        }
        p
    }

    /// The unit lower triangular factor.
    pub fn l(&self) -> Matrix {
        let n = self.order();
        Matrix::from_fn(n, n, |i, j| {
            if i > j {
                self.packed[(i, j)]
            } else if i == j {
                1.0
            } else {
                0.0
            }
        })
    }

    /// The upper triangular factor.
    pub fn u(&self) -> Matrix {
        let n = self.order();
        Matrix::from_fn(n, n, |i, j| if i <= j { self.packed[(i, j)] } else { 0.0 })
    }

    /// The determinant: the product of the pivots, with the sign of the permutation.
    pub fn determinant(&self) -> f64 {
        let mut det = self.sign;
        for i in 0..self.order() {
            det *= self.packed[(i, i)];
        }
        det
    }

    /// Solves `A * x = b`.
    pub fn solve(&self, b: &Vector) -> Result<Vector> {
        self.check_rhs(b.len(), "LU solve")?;
        let n = self.order();
        // Forward substitution on the permuted right-hand side; L has a unit diagonal.
        let mut x: Vec<f64> = self.permutation.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            let row = self.packed.row(i);
            let mut value = x[i];
            for j in 0..i {
                value -= row[j] * x[j];
            }
            x[i] = value;
        }
        for i in (0..n).rev() {
            let row = self.packed.row(i);
            let mut value = x[i];
            for j in (i + 1)..n {
                value -= row[j] * x[j];
            }
            x[i] = value / row[i];
        }
        Ok(Vector::new(x))
    }

    /// Solves the transposed system `A^T * x = b` with the same factors.
    pub fn solve_transpose(&self, b: &Vector) -> Result<Vector> {
        self.check_rhs(b.len(), "LU transpose solve")?;
        let n = self.order();
        // A^T = U^T * L^T * P, so solve with U^T, then with L^T, then undo the permutation.
        let mut y = b.as_slice().to_vec();
        for i in 0..n {
            let mut value = y[i];
            for j in 0..i {
                value -= self.packed[(j, i)] * y[j];
            }
            y[i] = value / self.packed[(i, i)];
        }
        for i in (0..n).rev() {
            let mut value = y[i];
            for j in (i + 1)..n {
                value -= self.packed[(j, i)] * y[j];
            }
            y[i] = value;
        }
        let mut x = vec![0.0; n];
        for (i, &p) in self.permutation.iter().enumerate() {
            x[p] = y[i];
        }
        Ok(Vector::new(x))
    }

    /// Solves `A * X = B` for a matrix of right-hand sides, one column at a time.
    pub fn solve_matrix(&self, b: &Matrix) -> Result<Matrix> {
        self.check_rhs(b.rows(), "LU solve")?;
        let mut x = Matrix::zeros(b.rows(), b.cols());
        for j in 0..b.cols() {
            let column = self.solve(&Vector::new(b.column(j)))?;
            for i in 0..b.rows() {
                x[(i, j)] = column[i];
            }
        }
        Ok(x)
    }

    /// The inverse of the decomposed matrix.
    pub fn inverse(&self) -> Result<Matrix> {
        self.solve_matrix(&Matrix::identity(self.order()))
    }

    fn check_rhs(&self, len: usize, op: &'static str) -> Result<()> {
        let n = self.order();
        if len != n {
            return Err(LinalgError::DimensionMismatch {
                op,
                left: (n, n),
                right: (len, 1),
            });
        }
        match self.singular_pivot {
            Some(pivot) => Err(LinalgError::Singular { pivot }),
            None => Ok(()),
        }
    }
}

impl Matrix {
    /// The LU decomposition with the default tolerance; see [`Lu::new`].
    pub fn lu(&self) -> Result<Lu> {
        Lu::new(self)
    }

    /// The determinant of a square matrix, computed from its LU decomposition.
    pub fn determinant(&self) -> Result<f64> {
        Ok(Lu::new(self)?.determinant())
    }

    /// The inverse of a square matrix, or [`LinalgError::Singular`].
    pub fn inverse(&self) -> Result<Matrix> {
        Lu::new(self)?.inverse()
    }

    /// Solves `self * x = b` by LU decomposition with partial pivoting.
    pub fn solve(&self, b: &Vector) -> Result<Vector> {
        Lu::new(self)?.solve(b)
    }
}
