//! Helpers shared by the integration tests.
#![allow(dead_code)]

use matrixlab::{Matrix, Vector};

/// Asserts that two numbers differ by at most `tol`.
#[track_caller]
pub fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {:e}, got {:e} (difference {:e}, tolerance {:e})",
        expected,
        actual,
        (actual - expected).abs(),
        tol
    );
}

/// Asserts that two matrices have the same shape and differ by at most `tol` element-wise.
#[track_caller]
pub fn assert_matrix_close(actual: &Matrix, expected: &Matrix, tol: f64) {
    assert_eq!(actual.shape(), expected.shape(), "shapes differ");
    for i in 0..actual.rows() {
        for j in 0..actual.cols() {
            let (a, e) = (actual[(i, j)], expected[(i, j)]);
            assert!(
                (a - e).abs() <= tol,
                "element ({}, {}): expected {:e}, got {:e} (tolerance {:e})",
                i,
                j,
                e,
                a,
                tol
            );
        }
    }
}

/// Asserts that a vector matches the expected elements to within `tol`.
#[track_caller]
pub fn assert_vector_close(actual: &Vector, expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "lengths differ");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "element {}: expected {:e}, got {:e} (tolerance {:e})",
            i,
            e,
            a,
            tol
        );
    }
}

/// The `n` x `n` Hilbert matrix, `h[i][j] = 1 / (i + j + 1)`: symmetric, positive definite
/// and famously ill-conditioned.
pub fn hilbert(n: usize) -> Matrix {
    Matrix::from_fn(n, n, |i, j| 1.0 / (i + j + 1) as f64)
}
