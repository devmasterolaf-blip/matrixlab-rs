mod common;

use common::{assert_close, assert_matrix_close, assert_vector_close, hilbert};
use matrixlab::{LinalgError, Matrix, Vector};

/// A textbook example with an integer factor: L = [2 0 0; 6 1 0; -8 5 3].
fn spd() -> Matrix {
    Matrix::from_rows(&[
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ])
    .unwrap()
}

#[test]
fn factor_matches_the_known_integer_result() {
    let chol = spd().cholesky().unwrap();
    let expected =
        Matrix::from_rows(&[[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]]).unwrap();
    assert_matrix_close(chol.l(), &expected, 1e-13);
}

#[test]
fn factor_times_its_transpose_gives_the_matrix_back() {
    let chol = spd().cholesky().unwrap();
    let product = chol.l().try_mul(&chol.l().transpose()).unwrap();
    assert_matrix_close(&product, &spd(), 1e-12);
}

#[test]
fn determinant_and_its_logarithm() {
    let chol = spd().cholesky().unwrap();
    // (2 * 1 * 3)^2
    assert_close(chol.determinant(), 36.0, 1e-11);
    assert_close(chol.log_determinant(), 36.0_f64.ln(), 1e-12);
}

#[test]
fn log_determinant_survives_where_the_determinant_underflows() {
    let tiny = Matrix::from_diagonal(&[1e-100; 5]);
    let chol = tiny.cholesky().unwrap();
    assert_eq!(chol.determinant(), 0.0);
    assert_close(chol.log_determinant(), 5.0 * 1e-100_f64.ln(), 1e-9);
}

#[test]
fn solve_recovers_a_known_solution() {
    // b = A * (1, 1, 1)
    let b = Vector::from_slice(&[0.0, 6.0, 39.0]);
    let x = spd().cholesky().unwrap().solve(&b).unwrap();
    assert_vector_close(&x, &[1.0, 1.0, 1.0], 1e-11);
}

#[test]
fn inverse_agrees_with_the_lu_inverse() {
    let via_cholesky = spd().cholesky().unwrap().inverse().unwrap();
    let via_lu = spd().inverse().unwrap();
    assert_matrix_close(&via_cholesky, &via_lu, 1e-10);
}

#[test]
fn hilbert_matrix_is_positive_definite_and_solvable() {
    let h = hilbert(6);
    let x_true = Vector::filled(6, 1.0);
    let b = h.try_mul_vec(&x_true).unwrap();
    let x = h.cholesky().unwrap().solve(&b).unwrap();
    // The condition number is about 1.5e7, so roughly nine digits survive.
    assert_vector_close(&x, x_true.as_slice(), 1e-8);
}

#[test]
fn an_indefinite_matrix_fails_at_the_offending_pivot() {
    let a = Matrix::from_rows(&[[1.0, 2.0], [2.0, 1.0]]).unwrap();
    assert_eq!(
        a.cholesky().unwrap_err(),
        LinalgError::NotPositiveDefinite { pivot: 1 }
    );
    let negative = Matrix::from_diagonal(&[-1.0, 2.0]);
    assert_eq!(
        negative.cholesky().unwrap_err(),
        LinalgError::NotPositiveDefinite { pivot: 0 }
    );
}

#[test]
fn a_semi_definite_matrix_is_rejected() {
    let a = Matrix::from_rows(&[[1.0, 1.0], [1.0, 1.0]]).unwrap();
    assert_eq!(
        a.cholesky().unwrap_err(),
        LinalgError::NotPositiveDefinite { pivot: 1 }
    );
}

#[test]
fn an_asymmetric_matrix_is_rejected_before_any_arithmetic() {
    let a = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(
        a.cholesky().unwrap_err(),
        LinalgError::NotSymmetric { row: 0, col: 1 }
    );
}

#[test]
fn rounding_level_asymmetry_is_tolerated() {
    let mut a = spd();
    a[(0, 1)] += 1e-13;
    assert!(a.cholesky().is_ok());
}

#[test]
fn shape_errors() {
    assert_eq!(
        Matrix::zeros(2, 3).cholesky().unwrap_err(),
        LinalgError::NotSquare { rows: 2, cols: 3 }
    );
    assert!(matches!(
        Matrix::zeros(0, 0).cholesky().unwrap_err(),
        LinalgError::Empty { .. }
    ));
    let err = spd().cholesky().unwrap().solve(&Vector::zeros(2)).unwrap_err();
    assert!(matches!(err, LinalgError::DimensionMismatch { .. }));
}
