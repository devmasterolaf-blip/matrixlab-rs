mod common;

use common::{assert_close, assert_matrix_close, assert_vector_close, hilbert};
use matrixlab::lu::Lu;
use matrixlab::{LinalgError, Matrix, Vector};

fn three_by_three() -> Matrix {
    Matrix::from_rows(&[[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]]).unwrap()
}

#[test]
fn factors_reproduce_the_permuted_matrix() {
    let a = three_by_three();
    let lu = a.lu().unwrap();
    let pa = lu.permutation_matrix().try_mul(&a).unwrap();
    let product = lu.l().try_mul(&lu.u()).unwrap();
    assert_matrix_close(&product, &pa, 1e-14);
    // Partial pivoting brings the 4 to the top.
    assert_eq!(lu.permutation()[0], 1);
    assert!(!lu.is_singular());
}

#[test]
fn l_is_unit_lower_and_u_is_upper_triangular() {
    let lu = three_by_three().lu().unwrap();
    let (l, u) = (lu.l(), lu.u());
    for i in 0..3 {
        assert_eq!(l[(i, i)], 1.0);
        for j in 0..3 {
            if j > i {
                assert_eq!(l[(i, j)], 0.0);
            }
            if j < i {
                assert_eq!(u[(i, j)], 0.0);
                assert!(l[(i, j)].abs() <= 1.0, "multipliers are bounded by one");
            }
        }
    }
}

#[test]
fn determinant_of_a_hand_computed_case() {
    // 2 * (-12 - 0) - 1 * (8 - 0) + 1 * (28 - 12) = -16
    assert_close(three_by_three().determinant().unwrap(), -16.0, 1e-13);
}

#[test]
fn determinant_tracks_the_sign_of_row_swaps() {
    let swap = Matrix::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap();
    assert_eq!(swap.determinant().unwrap(), -1.0);
    let cycle =
        Matrix::from_rows(&[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).unwrap();
    assert_eq!(cycle.determinant().unwrap(), 1.0);
}

#[test]
fn solve_recovers_a_known_solution() {
    // b = A * (1, 2, 3)
    let b = Vector::from_slice(&[7.0, -8.0, 18.0]);
    let x = three_by_three().solve(&b).unwrap();
    assert_vector_close(&x, &[1.0, 2.0, 3.0], 1e-14);
}

#[test]
fn a_tiny_leading_element_is_handled_by_pivoting() {
    // Without pivoting the 1e-20 pivot would wipe out the second equation.
    let a = Matrix::from_rows(&[[1e-20, 1.0], [1.0, 1.0]]).unwrap();
    let x = a.solve(&Vector::from_slice(&[1.0, 2.0])).unwrap();
    assert_vector_close(&x, &[1.0, 1.0], 1e-15);
}

#[test]
fn inverse_of_a_two_by_two() {
    let a = Matrix::from_rows(&[[4.0, 7.0], [2.0, 6.0]]).unwrap();
    let expected = Matrix::from_rows(&[[0.6, -0.7], [-0.2, 0.4]]).unwrap();
    assert_matrix_close(&a.inverse().unwrap(), &expected, 1e-15);
}

#[test]
fn inverse_of_the_hilbert_matrix_has_integer_entries() {
    let expected = Matrix::from_rows(&[
        [9.0, -36.0, 30.0],
        [-36.0, 192.0, -180.0],
        [30.0, -180.0, 180.0],
    ])
    .unwrap();
    assert_matrix_close(&hilbert(3).inverse().unwrap(), &expected, 1e-11);
}

#[test]
fn determinant_of_the_five_by_five_hilbert_matrix() {
    // The exact value is 1 / 266716800000.
    let det = hilbert(5).determinant().unwrap();
    let exact = 1.0 / 266_716_800_000.0;
    assert_close(det, exact, 1e-10 * exact);
}

#[test]
fn solve_matrix_handles_several_right_hand_sides() {
    let a = three_by_three();
    let x_true = Matrix::from_rows(&[[1.0, 0.0], [2.0, -1.0], [3.0, 4.0]]).unwrap();
    let b = a.try_mul(&x_true).unwrap();
    let x = a.lu().unwrap().solve_matrix(&b).unwrap();
    assert_matrix_close(&x, &x_true, 1e-13);
}

#[test]
fn transpose_solve_agrees_with_decomposing_the_transpose() {
    let a = three_by_three();
    let b = Vector::from_slice(&[1.0, -2.0, 0.5]);
    let via_factors = a.lu().unwrap().solve_transpose(&b).unwrap();
    let direct = a.transpose().solve(&b).unwrap();
    assert_vector_close(&via_factors, direct.as_slice(), 1e-13);
}

#[test]
fn an_exactly_singular_matrix_is_detected() {
    let a = Matrix::from_rows(&[[1.0, 2.0], [2.0, 4.0]]).unwrap();
    let lu = a.lu().unwrap();
    assert!(lu.is_singular());
    assert_eq!(lu.singular_pivot(), Some(1));
    assert_eq!(lu.determinant(), 0.0);
    assert_eq!(
        a.solve(&Vector::from_slice(&[1.0, 1.0])),
        Err(LinalgError::Singular { pivot: 1 })
    );
    assert_eq!(a.inverse(), Err(LinalgError::Singular { pivot: 1 }));
}

#[test]
fn a_zero_column_is_singular_at_that_column() {
    let a = Matrix::from_rows(&[[0.0, 1.0, 2.0], [0.0, 3.0, 4.0], [0.0, 5.0, 6.0]]).unwrap();
    let lu = a.lu().unwrap();
    assert_eq!(lu.singular_pivot(), Some(0));
    assert_eq!(lu.determinant(), 0.0);
}

#[test]
fn the_tolerance_decides_what_counts_as_singular() {
    // After elimination the second pivot is about 1e-10.
    let a = Matrix::from_rows(&[[1.0, 1.0], [1.0, 1.0 + 1e-10]]).unwrap();
    assert!(!Lu::new(&a).unwrap().is_singular());
    assert!(Lu::with_tolerance(&a, 1e-8).unwrap().is_singular());
    assert!(!Lu::with_tolerance(&a, 0.0).unwrap().is_singular());
}

#[test]
fn rank_one_perturbation_below_rounding_is_singular() {
    // 1 + 1e-17 rounds to 1, so the rows are identical in floating point.
    let a = Matrix::from_rows(&[[1.0, 1.0], [1.0, 1.0 + 1e-17]]).unwrap();
    assert!(a.lu().unwrap().is_singular());
}

#[test]
fn invalid_inputs_are_rejected() {
    assert_eq!(
        Matrix::zeros(2, 3).lu().unwrap_err(),
        LinalgError::NotSquare { rows: 2, cols: 3 }
    );
    assert_eq!(
        Matrix::zeros(0, 0).lu().unwrap_err(),
        LinalgError::Empty {
            op: "LU decomposition"
        }
    );
    assert!(matches!(
        Lu::with_tolerance(&Matrix::identity(2), -1.0),
        Err(LinalgError::InvalidArgument { .. })
    ));
    assert!(matches!(
        Lu::with_tolerance(&Matrix::identity(2), f64::NAN),
        Err(LinalgError::InvalidArgument { .. })
    ));
}

#[test]
fn right_hand_side_of_the_wrong_length_is_a_dimension_error() {
    let err = three_by_three().solve(&Vector::zeros(2)).unwrap_err();
    assert_eq!(
        err,
        LinalgError::DimensionMismatch {
            op: "LU solve",
            left: (3, 3),
            right: (2, 1)
        }
    );
}

#[test]
fn a_nan_entry_makes_the_matrix_singular_instead_of_poisoning_the_result() {
    let a = Matrix::from_rows(&[[f64::NAN, 1.0], [1.0, 1.0]]).unwrap();
    assert!(a.lu().unwrap().is_singular());
}
