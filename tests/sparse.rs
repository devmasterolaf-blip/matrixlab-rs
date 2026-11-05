use matrixlab::sparse::CooMatrix;
use matrixlab::{LinalgError, Matrix, Vector};

/// [ 4 0 1 ]
/// [ 0 3 0 ]
/// [ 2 0 5 ]
fn dense() -> Matrix {
    Matrix::from_rows(&[[4.0, 0.0, 1.0], [0.0, 3.0, 0.0], [2.0, 0.0, 5.0]]).unwrap()
}

fn coo() -> CooMatrix {
    // Deliberately out of order.
    CooMatrix::from_triplets(
        3,
        3,
        &[
            (2, 2, 5.0),
            (0, 0, 4.0),
            (1, 1, 3.0),
            (2, 0, 2.0),
            (0, 2, 1.0),
        ],
    )
    .unwrap()
}

#[test]
fn coo_reports_its_shape_and_entry_count() {
    let m = coo();
    assert_eq!(m.shape(), (3, 3));
    assert_eq!(m.nnz(), 5);
    assert_eq!(m.iter().next(), Some((2, 2, 5.0)));
}

#[test]
fn coo_converts_to_dense() {
    assert_eq!(coo().to_dense(), dense());
}

#[test]
fn coo_from_dense_skips_zeros() {
    let m = CooMatrix::from_dense(&dense());
    assert_eq!(m.nnz(), 5);
    assert_eq!(m.to_dense(), dense());
}

#[test]
fn coo_duplicates_add_up() {
    let mut m = CooMatrix::new(2, 2);
    m.push(0, 1, 1.5).unwrap();
    m.push(0, 1, 2.5).unwrap();
    m.push(1, 0, -1.0).unwrap();
    assert_eq!(m.nnz(), 3);
    assert_eq!(
        m.to_dense(),
        Matrix::from_rows(&[[0.0, 4.0], [-1.0, 0.0]]).unwrap()
    );
}

#[test]
fn coo_push_checks_both_indices() {
    let mut m = CooMatrix::new(2, 3);
    assert_eq!(
        m.push(2, 0, 1.0),
        Err(LinalgError::IndexOutOfBounds {
            index: (2, 0),
            shape: (2, 3)
        })
    );
    assert!(m.push(0, 3, 1.0).is_err());
    assert!(m.push(1, 2, 1.0).is_ok());
    assert!(CooMatrix::from_triplets(1, 1, &[(0, 0, 1.0), (1, 0, 2.0)]).is_err());
}

#[test]
fn coo_transpose_swaps_indices_and_shape() {
    let m = CooMatrix::from_triplets(2, 3, &[(0, 2, 7.0), (1, 0, -1.0)]).unwrap();
    let t = m.transpose();
    assert_eq!(t.shape(), (3, 2));
    assert_eq!(t.to_dense(), m.to_dense().transpose());
}

#[test]
fn coo_matrix_vector_product_matches_the_dense_one() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let expected = dense().try_mul_vec(&x).unwrap();
    assert_eq!(coo().mul_vec(&x).unwrap(), expected);
    assert_eq!(expected.as_slice(), &[7.0, 6.0, 17.0]);
    assert!(matches!(
        coo().mul_vec(&Vector::zeros(2)),
        Err(LinalgError::DimensionMismatch { .. })
    ));
}
