use matrixlab::Matrix;

fn m() -> Matrix {
    Matrix::from_rows(&[[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]).unwrap()
}

#[test]
fn one_norm_is_the_largest_column_sum() {
    // column sums of absolute values: 5, 7, 9
    assert_eq!(m().norm_one(), 9.0);
}

#[test]
fn infinity_norm_is_the_largest_row_sum() {
    // row sums of absolute values: 6, 15
    assert_eq!(m().norm_inf(), 15.0);
}

#[test]
fn frobenius_norm_is_the_root_of_the_sum_of_squares() {
    // 1 + 4 + 9 + 16 + 25 + 36 = 91
    assert!((m().norm_frobenius() - 91.0_f64.sqrt()).abs() < 1e-14);
}

#[test]
fn max_norm_is_the_largest_magnitude() {
    assert_eq!(m().norm_max(), 6.0);
}

#[test]
fn one_and_infinity_norms_swap_under_transposition() {
    let t = m().transpose();
    assert_eq!(t.norm_one(), m().norm_inf());
    assert_eq!(t.norm_inf(), m().norm_one());
}

#[test]
fn norms_of_an_empty_matrix_are_zero() {
    let e = Matrix::zeros(0, 3);
    assert_eq!(e.norm_one(), 0.0);
    assert_eq!(e.norm_inf(), 0.0);
    assert_eq!(e.norm_frobenius(), 0.0);
    assert_eq!(e.norm_max(), 0.0);
}

#[test]
fn norms_of_the_identity() {
    let i = Matrix::identity(4);
    assert_eq!(i.norm_one(), 1.0);
    assert_eq!(i.norm_inf(), 1.0);
    assert_eq!(i.norm_frobenius(), 2.0);
}

#[test]
fn frobenius_norm_of_huge_elements_does_not_overflow() {
    let big = Matrix::filled(2, 2, 1e200);
    assert!((big.norm_frobenius() - 2e200).abs() < 1e186);
}

#[test]
fn a_nan_element_makes_every_norm_nan() {
    let mut bad = m();
    bad[(1, 1)] = f64::NAN;
    assert!(bad.norm_one().is_nan());
    assert!(bad.norm_inf().is_nan());
    assert!(bad.norm_frobenius().is_nan());
    assert!(bad.norm_max().is_nan());
}
