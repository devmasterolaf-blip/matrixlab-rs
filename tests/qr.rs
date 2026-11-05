mod common;

use common::{assert_close, assert_matrix_close, assert_vector_close, hilbert};
use matrixlab::qr::lstsq;
use matrixlab::{LinalgError, Matrix, Vector};

fn tall() -> Matrix {
    Matrix::from_rows(&[[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]).unwrap()
}

#[test]
fn q_times_r_reproduces_the_matrix() {
    let a = tall();
    let qr = a.qr().unwrap();
    assert_eq!(qr.shape(), (4, 2));
    assert_matrix_close(&qr.q().try_mul(&qr.r()).unwrap(), &a, 1e-14);
}

#[test]
fn q_has_orthonormal_columns() {
    let q = tall().qr().unwrap().q();
    assert_eq!(q.shape(), (4, 2));
    let gram = q.transpose().try_mul(&q).unwrap();
    assert_matrix_close(&gram, &Matrix::identity(2), 1e-14);
}

#[test]
fn r_is_upper_triangular_with_the_column_norm_in_the_corner() {
    let r = tall().qr().unwrap().r();
    assert_eq!(r.shape(), (2, 2));
    assert_eq!(r[(1, 0)], 0.0);
    // The first column of A is (1, 1, 1, 1), whose length is 2.
    assert_close(r[(0, 0)].abs(), 2.0, 1e-14);
}

#[test]
fn an_exact_line_is_fitted_with_zero_residual() {
    // y = 1 + 2 x sampled at x = 0, 1, 2, 3
    let a = Matrix::from_rows(&[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]).unwrap();
    let b = Vector::from_slice(&[1.0, 3.0, 5.0, 7.0]);
    let fit = lstsq(&a, &b).unwrap();
    assert_vector_close(&fit.solution, &[1.0, 2.0], 1e-13);
    assert_close(fit.residual_norm, 0.0, 1e-13);
}

#[test]
fn a_noisy_line_gives_the_normal_equations_solution() {
    // Points (1, 6), (2, 5), (3, 7), (4, 10): the best line is y = 3.5 + 1.4 x and the
    // residuals are 1.1, -1.3, -0.7, 0.9, whose squares add up to 4.2.
    let b = Vector::from_slice(&[6.0, 5.0, 7.0, 10.0]);
    let fit = lstsq(&tall(), &b).unwrap();
    assert_vector_close(&fit.solution, &[3.5, 1.4], 1e-13);
    assert_close(fit.residual_norm, 4.2_f64.sqrt(), 1e-13);
}

#[test]
fn reported_residual_matches_the_recomputed_one() {
    let a = tall();
    let b = Vector::from_slice(&[2.0, -1.0, 0.5, 3.0]);
    let fit = lstsq(&a, &b).unwrap();
    let residual = a.try_mul_vec(&fit.solution).unwrap().try_sub(&b).unwrap();
    assert_close(fit.residual_norm, residual.norm2(), 1e-13);
    // At the minimum the residual is orthogonal to the columns of A.
    let gradient = a.transpose().try_mul_vec(&residual).unwrap();
    assert_close(gradient.norm_inf(), 0.0, 1e-12);
}

#[test]
fn a_square_system_is_solved_like_lu() {
    let a = Matrix::from_rows(&[[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]]).unwrap();
    let b = Vector::from_slice(&[7.0, -8.0, 18.0]);
    let fit = lstsq(&a, &b).unwrap();
    assert_vector_close(&fit.solution, &[1.0, 2.0, 3.0], 1e-13);
    assert_eq!(fit.residual_norm, 0.0);
}

#[test]
fn polynomial_fit_on_an_ill_conditioned_basis() {
    // Fit a cubic through seven samples of p(t) = 1 - t + 2 t^2 - 3 t^3 with a Vandermonde
    // matrix; QR keeps the accuracy that the normal equations would lose.
    let ts: [f64; 7] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    let a = Matrix::from_fn(7, 4, |i, j| ts[i].powi(j as i32));
    let b: Vector = ts
        .iter()
        .map(|&t| 1.0 - t + 2.0 * t * t - 3.0 * t * t * t)
        .collect();
    let fit = lstsq(&a, &b).unwrap();
    assert_vector_close(&fit.solution, &[1.0, -1.0, 2.0, -3.0], 1e-11);
    assert_close(fit.residual_norm, 0.0, 1e-11);
}

#[test]
fn hilbert_system_is_solved_to_the_accuracy_its_conditioning_allows() {
    let h = hilbert(5);
    let b = h.try_mul_vec(&Vector::filled(5, 1.0)).unwrap();
    let fit = lstsq(&h, &b).unwrap();
    // cond(H5) is about 4.8e5.
    assert_vector_close(&fit.solution, &[1.0; 5], 1e-9);
}

#[test]
fn dependent_columns_are_reported_as_rank_deficient() {
    let a = Matrix::from_rows(&[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]).unwrap();
    let err = lstsq(&a, &Vector::from_slice(&[1.0, 2.0, 3.0])).unwrap_err();
    assert_eq!(err, LinalgError::RankDeficient { column: 1 });
    assert_eq!(a.qr().unwrap().deficient_column(), Some(1));
}

#[test]
fn a_zero_column_is_rank_deficient_at_that_column() {
    let a = Matrix::from_rows(&[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]).unwrap();
    let qr = a.qr().unwrap();
    assert_eq!(qr.deficient_column(), Some(0));
    // The decomposition itself is still valid.
    assert_matrix_close(&qr.q().try_mul(&qr.r()).unwrap(), &a, 1e-14);
}

#[test]
fn shape_errors() {
    assert_eq!(
        Matrix::zeros(2, 3).qr().unwrap_err(),
        LinalgError::Underdetermined { rows: 2, cols: 3 }
    );
    assert!(matches!(
        Matrix::zeros(0, 0).qr().unwrap_err(),
        LinalgError::Empty { .. }
    ));
    let err = lstsq(&tall(), &Vector::zeros(3)).unwrap_err();
    assert_eq!(
        err,
        LinalgError::DimensionMismatch {
            op: "QR solve",
            left: (4, 2),
            right: (3, 1)
        }
    );
}
