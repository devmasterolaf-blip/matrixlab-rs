use matrixlab::{LinalgError, Matrix};

/// A 4 x 5 matrix whose element (i, j) is 10 * i + j, so every value names its position.
fn grid() -> Matrix {
    Matrix::from_fn(4, 5, |i, j| (10 * i + j) as f64)
}

#[test]
fn a_block_view_reads_the_right_elements() {
    let m = grid();
    let v = m.view(1..3, 2..5).unwrap();
    assert_eq!(v.shape(), (2, 3));
    assert_eq!(v[(0, 0)], 12.0);
    assert_eq!(v[(1, 2)], 24.0);
    assert_eq!(v.get(1, 2), Some(24.0));
    assert_eq!(v.get(2, 0), None);
    assert_eq!(v.row(1), &[22.0, 23.0, 24.0]);
    assert_eq!(v.column(1), vec![13.0, 23.0]);
}

#[test]
fn views_copy_out_to_a_matrix() {
    let m = grid();
    let block = m.submatrix(0..2, 3..5).unwrap();
    assert_eq!(block, Matrix::from_rows(&[[3.0, 4.0], [13.0, 14.0]]).unwrap());
    assert_eq!(m.as_view().to_matrix(), m);
}

#[test]
fn row_and_column_views_convert_to_vectors() {
    let m = grid();
    assert_eq!(
        m.row_view(2).unwrap().to_vector().unwrap().as_slice(),
        &[20.0, 21.0, 22.0, 23.0, 24.0]
    );
    assert_eq!(
        m.column_view(4).unwrap().to_vector().unwrap().as_slice(),
        &[4.0, 14.0, 24.0, 34.0]
    );
}

#[test]
fn a_block_is_not_a_vector() {
    let m = grid();
    let err = m.view(0..2, 0..2).unwrap().to_vector().unwrap_err();
    assert!(matches!(err, LinalgError::DimensionMismatch { .. }));
}

#[test]
fn nested_views_are_relative_to_the_outer_view() {
    let m = grid();
    let outer = m.view(1..4, 1..5).unwrap();
    let inner = outer.view(1..3, 2..4).unwrap();
    assert_eq!(inner.to_matrix().as_slice(), &[23.0, 24.0, 33.0, 34.0]);
    // The inner ranges are checked against the outer view, not the whole matrix.
    assert_eq!(
        outer.view(0..4, 0..1).unwrap_err(),
        LinalgError::IndexOutOfBounds {
            index: (4, 1),
            shape: (3, 4)
        }
    );
}

#[test]
fn out_of_range_and_backward_ranges_are_rejected() {
    let m = grid();
    assert_eq!(
        m.view(0..5, 0..1).unwrap_err(),
        LinalgError::IndexOutOfBounds {
            index: (5, 1),
            shape: (4, 5)
        }
    );
    #[allow(clippy::reversed_empty_ranges)]
    let backwards = m.view(3..1, 0..1);
    assert!(matches!(
        backwards.unwrap_err(),
        LinalgError::InvalidArgument { .. }
    ));
    assert!(m.row_view(4).is_err());
    assert!(m.column_view(5).is_err());
}

#[test]
fn empty_views_are_allowed() {
    let m = grid();
    let v = m.view(2..2, 0..5).unwrap();
    assert_eq!(v.shape(), (0, 5));
    assert_eq!(v.to_matrix().shape(), (0, 5));
    assert_eq!(v.sum(), 0.0);
}

#[test]
fn sum_and_norm_of_a_view() {
    let m = grid();
    let v = m.view(0..2, 0..2).unwrap();
    // elements 0, 1, 10, 11
    assert_eq!(v.sum(), 22.0);
    assert!((v.norm_frobenius() - 222.0_f64.sqrt()).abs() < 1e-13);
}

#[test]
fn set_block_overwrites_a_window() {
    let mut m = Matrix::zeros(3, 3);
    let block = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    m.set_block(1, 1, &block).unwrap();
    assert_eq!(
        m.as_slice(),
        &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0]
    );
    assert_eq!(
        m.set_block(2, 2, &block),
        Err(LinalgError::IndexOutOfBounds {
            index: (4, 4),
            shape: (3, 3)
        })
    );
}
