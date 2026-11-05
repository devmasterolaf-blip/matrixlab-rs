//! Compensated summation.
//!
//! Adding many floating-point numbers of mixed size loses the low bits of the small ones.
//! The accumulator here keeps a running correction term (Neumaier's variant of Kahan
//! summation), which brings the error of a sum down to a few units in the last place of the
//! result for all but pathological inputs. It is used for dot products, norms and the
//! residuals of the solvers.

/// A running sum with a correction term.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    /// An empty sum.
    pub fn new() -> CompensatedSum {
        CompensatedSum::default()
    }

    /// Adds one term.
    pub fn add(&mut self, value: f64) {
        let t = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - t) + value;
        } else {
            self.correction += (value - t) + self.sum;
        }
        self.sum = t;
    }

    /// The sum of everything added so far.
    pub fn value(&self) -> f64 {
        self.sum + self.correction
    }
}

/// The compensated sum of a sequence of numbers.
pub fn sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut acc = CompensatedSum::new();
    for v in values {
        acc.add(v);
    }
    acc.value()
}

/// The compensated dot product of two slices.
///
/// The slices must have the same length; the callers in this crate check that and report a
/// typed error. When they differ the extra elements of the longer one are ignored.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    sum(a.iter().zip(b.iter()).map(|(&x, &y)| x * y))
}

/// The Euclidean length of a slice, computed without overflow or underflow of the squares.
///
/// The elements are divided by the largest magnitude before squaring, so a vector whose
/// elements are around `1e200` or `1e-200` still gets the right norm.
pub fn norm2(values: &[f64]) -> f64 {
    let scale = values.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    if scale == 0.0 || !scale.is_finite() {
        // All zeros, or an infinity or NaN among the elements: the maximum (or the NaN that
        // the plain sum of squares produces) is the answer.
        return if values.iter().any(|x| x.is_nan()) {
            f64::NAN
        } else {
            scale
        };
    }
    let ssq = sum(values.iter().map(|&x| {
        let y = x / scale;
        y * y
    }));
    scale * ssq.sqrt()
}
