mod common;

use common::{assert_close, hilbert};
use matrixlab::{LinalgError, Matrix};

#[test]
fn rank_of_simple_matrices() {
    assert_eq!(Matrix::identity(3).rank(), 3);
    assert_eq!(Matrix::zeros(3, 4).rank(), 0);
    assert_eq!(Matrix::zeros(0, 0).rank(), 0);
    let singular = Matrix::from_rows(&[[1.0, 2.0], [2.0, 4.0]]).unwrap();
    assert_eq!(singular.rank(), 1);
}

#[test]
fn rank_of_the_three_by_three_counting_matrix_is_two() {
    // Row 3 = 2 * row 2 - row 1.
    let a = Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();
    assert_eq!(a.rank(), 2);
}

#[test]
fn rank_of_rectangular_matrices() {
    let wide = Matrix::from_rows(&[[1.0, 0.0, 2.0, 1.0], [0.0, 1.0, 1.0, 3.0]]).unwrap();
    assert_eq!(wide.rank(), 2);
    assert_eq!(wide.transpose().rank(), 2);
    let dependent = Matrix::from_rows(&[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]).unwrap();
    assert_eq!(dependent.rank(), 1);
}

#[test]
fn rank_needs_column_pivoting_when_the_leading_columns_are_zero() {
    let a = Matrix::from_rows(&[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 3.0, 0.0]]).unwrap();
    assert_eq!(a.rank(), 2);
}

#[test]
fn outer_product_has_rank_one() {
    let u = [1.0, -2.0, 3.0, 0.5, 4.0, -1.5, 2.5];
    let v = [0.3, 0.7, -1.1, 2.0, 1.0, 0.1, -0.9];
    let a = Matrix::from_fn(7, 7, |i, j| u[i] * v[j]);
    assert_eq!(a.rank(), 1);
}

#[test]
fn the_tolerance_changes_the_numerical_rank_of_an_ill_conditioned_matrix() {
    let h = hilbert(8);
    assert_eq!(h.rank(), 8);
    assert!(h.rank_with_tolerance(1e-6).unwrap() < 8);
    assert_eq!(h.rank_with_tolerance(0.0).unwrap(), 8);
    assert!(matches!(
        h.rank_with_tolerance(-1.0),
        Err(LinalgError::InvalidArgument { .. })
    ));
}

#[test]
fn condition_number_of_a_hand_computed_case() {
    // ||A||_1 = 6; the inverse is [-2 1; 1.5 -0.5] with 1-norm 3.5.
    let a = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    assert_close(a.condition_number().unwrap(), 21.0, 1e-12);
    assert_close(a.condition_estimate().unwrap(), 21.0, 1e-12);
}

#[test]
fn condition_number_of_the_identity_and_of_a_scaled_identity_is_one() {
    assert_eq!(Matrix::identity(4).condition_number().unwrap(), 1.0);
    assert_eq!(Matrix::identity(4).scale(1e-8).condition_estimate().unwrap(), 1.0);
}

#[test]
fn condition_numbers_of_hilbert_matrices_match_the_published_values() {
    // The 1-norm condition numbers of the Hilbert matrices of order 3, 4 and 5.
    for (n, expected) in [(3, 748.0), (4, 28_375.0), (5, 943_656.0)] {
        let cond = hilbert(n).condition_number().unwrap();
        assert_close(cond, expected, 1e-7 * expected);
    }
}

#[test]
fn the_estimate_is_a_lower_bound_within_a_small_factor() {
    for n in 2..=8 {
        let h = hilbert(n);
        let exact = h.condition_number().unwrap();
        let estimate = h.condition_estimate().unwrap();
        assert!(estimate <= exact * (1.0 + 1e-6), "n = {}", n);
        assert!(estimate >= exact / 3.0, "n = {}", n);
    }
}

#[test]
fn a_singular_matrix_has_an_infinite_condition_number() {
    let a = Matrix::from_rows(&[[1.0, 2.0], [2.0, 4.0]]).unwrap();
    assert_eq!(a.condition_number().unwrap(), f64::INFINITY);
    assert_eq!(a.condition_estimate().unwrap(), f64::INFINITY);
}

#[test]
fn condition_number_needs_a_square_matrix() {
    assert_eq!(
        Matrix::zeros(2, 3).condition_number().unwrap_err(),
        LinalgError::NotSquare { rows: 2, cols: 3 }
    );
    assert!(Matrix::zeros(2, 3).condition_estimate().is_err());
}
