//! Matrix norms.

use crate::matrix::Matrix;
use crate::sum;

impl Matrix {
    /// The 1-norm: the largest absolute column sum.
    pub fn norm_one(&self) -> f64 {
        let mut sums = vec![sum::CompensatedSum::new(); self.cols()];
        for row in self.iter_rows() {
            for (acc, &x) in sums.iter_mut().zip(row) {
                acc.add(x.abs());
            }
        }
        largest(sums.iter().map(|acc| acc.value()))
    }

    /// The infinity norm: the largest absolute row sum.
    pub fn norm_inf(&self) -> f64 {
        largest(
            self.iter_rows()
                .map(|row| sum::sum(row.iter().map(|x| x.abs()))),
        )
    }

    /// The Frobenius norm: the square root of the sum of the squared elements, computed
    /// with scaling so the squares cannot overflow.
    pub fn norm_frobenius(&self) -> f64 {
        sum::norm2(self.as_slice())
    }

    /// The largest absolute value of any element.
    pub fn norm_max(&self) -> f64 {
        largest(self.as_slice().iter().map(|x| x.abs()))
    }
}

/// The maximum of a sequence of non-negative numbers; zero when it is empty and NaN when
/// any of them is NaN.
fn largest<I: Iterator<Item = f64>>(values: I) -> f64 {
    let mut max = 0.0_f64;
    for v in values {
        if v.is_nan() {
            return f64::NAN;
        }
        max = max.max(v);
    }
    max
}
