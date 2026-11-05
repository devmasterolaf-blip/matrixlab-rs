//! Rank and condition numbers.

use crate::error::{LinalgError, Result};
use crate::lu::Lu;
use crate::matrix::Matrix;
use crate::vector::Vector;

impl Matrix {
    /// The numerical rank with the default tolerance, `max(rows, cols) * f64::EPSILON`.
    pub fn rank(&self) -> usize {
        let tolerance = self.rows().max(self.cols()) as f64 * f64::EPSILON;
        count_pivots(self, tolerance)
    }

    /// The numerical rank: the number of pivots of Gaussian elimination with complete
    /// pivoting whose magnitude exceeds `tolerance` times the largest magnitude in the
    /// matrix.
    ///
    /// Fails with [`LinalgError::InvalidArgument`] for a negative or NaN tolerance.
    pub fn rank_with_tolerance(&self, tolerance: f64) -> Result<usize> {
        if !(tolerance >= 0.0) {
            return Err(LinalgError::InvalidArgument {
                what: "the rank tolerance must be zero or positive",
            });
        }
        Ok(count_pivots(self, tolerance))
    }

    /// The condition number in the 1-norm, `||A||_1 * ||A^-1||_1`, from the explicit
    /// inverse. Returns infinity for a matrix that is singular to working precision.
    pub fn condition_number(&self) -> Result<f64> {
        let lu = Lu::new(self)?;
        if lu.is_singular() {
            return Ok(f64::INFINITY);
        }
        Ok(self.norm_one() * lu.inverse()?.norm_one())
    }

    /// An estimate of the 1-norm condition number that avoids forming the inverse.
    ///
    /// The norm of the inverse is estimated with Hager's method from a few solves with the
    /// LU factors, so the cost after the decomposition is `O(n^2)` instead of `O(n^3)`. The
    /// estimate never exceeds the true condition number and is usually within a small
    /// factor of it. Returns infinity for a singular matrix.
    pub fn condition_estimate(&self) -> Result<f64> {
        let lu = Lu::new(self)?;
        if lu.is_singular() {
            return Ok(f64::INFINITY);
        }
        Ok(self.norm_one() * inverse_norm_estimate(&lu)?)
    }
}

/// Counts the pivots larger than `tolerance * max|a|` in an elimination with complete
/// pivoting. Searching the whole remaining block for each pivot is what makes the count a
/// reliable rank: a negligible remainder means every remaining row is a combination of the
/// pivot rows.
fn count_pivots(a: &Matrix, tolerance: f64) -> usize {
    let (m, n) = a.shape();
    let threshold = tolerance * a.norm_max();
    let mut work = a.clone();
    let mut rank = 0;
    for k in 0..m.min(n) {
        let mut pivot = (k, k);
        let mut pivot_abs = 0.0;
        for i in k..m {
            for j in k..n {
                let candidate = work[(i, j)].abs();
                if candidate > pivot_abs {
                    pivot = (i, j);
                    pivot_abs = candidate;
                }
            }
        }
        if !(pivot_abs > threshold) || pivot_abs == 0.0 {
            break;
        }
        work.swap_rows(k, pivot.0);
        if pivot.1 != k {
            for i in 0..m {
                let tmp = work[(i, k)];
                work[(i, k)] = work[(i, pivot.1)];
                work[(i, pivot.1)] = tmp;
            }
        }
        let diagonal = work[(k, k)];
        for i in (k + 1)..m {
            let factor = work[(i, k)] / diagonal;
            if factor != 0.0 {
                for j in k..n {
                    let upper = work[(k, j)];
                    work[(i, j)] -= factor * upper;
                }
            }
        }
        rank += 1;
    }
    rank
}

/// Hager's estimate of `||A^-1||_1` from the LU factors of `A`.
///
/// The 1-norm of a matrix is the maximum of the convex function `||B * x||_1` over the unit
/// ball, attained at a unit vector. Each step solves once with `A` and once with `A^T` to
/// find the unit vector that most increases the function, and stops at a local maximum.
fn inverse_norm_estimate(lu: &Lu) -> Result<f64> {
    let n = lu.order();
    let mut x = Vector::filled(n, 1.0 / n as f64);
    let mut estimate = 0.0_f64;
    for _ in 0..5 {
        let y = lu.solve(&x)?;
        let y_norm = y.norm1();
        if y_norm <= estimate {
            break;
        }
        estimate = y_norm;
        let signs: Vector = y
            .iter()
            .map(|&v| if v >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        let z = lu.solve_transpose(&signs)?;
        let mut best = 0;
        for j in 1..n {
            if z[j].abs() > z[best].abs() {
                best = j;
            }
        }
        if z[best].abs() <= z.dot(&x)? {
            break;
        }
        x = Vector::unit(n, best);
    }
    Ok(estimate)
}
