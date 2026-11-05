use matrixlab::{LinalgError, Matrix};

#[test]
fn zeros_and_filled_have_the_requested_shape() {
    let z = Matrix::zeros(2, 3);
    assert_eq!(z.shape(), (2, 3));
    assert!(z.as_slice().iter().all(|&x| x == 0.0));
    let f = Matrix::filled(3, 1, 2.5);
    assert_eq!(f.as_slice(), &[2.5, 2.5, 2.5]);
    assert!(!f.is_square());
}

#[test]
fn identity_has_ones_only_on_the_diagonal() {
    let i = Matrix::identity(3);
    for r in 0..3 {
        for c in 0..3 {
            assert_eq!(i[(r, c)], if r == c { 1.0 } else { 0.0 });
        }
    }
}

#[test]
fn from_vec_is_row_major() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(m[(0, 2)], 3.0);
    assert_eq!(m[(1, 0)], 4.0);
    assert_eq!(m.into_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn from_vec_rejects_a_wrong_length() {
    let err = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0]).unwrap_err();
    assert_eq!(
        err,
        LinalgError::InvalidShape {
            rows: 2,
            cols: 2,
            len: 3
        }
    );
}

#[test]
fn from_rows_accepts_arrays_and_vectors() {
    let a = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    let b = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.shape(), (2, 2));
}

#[test]
fn from_rows_reports_the_first_ragged_row() {
    let err = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0], vec![4.0]]).unwrap_err();
    assert_eq!(
        err,
        LinalgError::RaggedRows {
            row: 1,
            expected: 2,
            found: 1
        }
    );
}

#[test]
fn from_rows_of_nothing_is_an_empty_matrix() {
    let rows: Vec<Vec<f64>> = Vec::new();
    let m = Matrix::from_rows(&rows).unwrap();
    assert_eq!(m.shape(), (0, 0));
    assert!(m.is_empty());
}

#[test]
fn from_fn_and_from_diagonal() {
    let m = Matrix::from_fn(2, 3, |i, j| (10 * i + j) as f64);
    assert_eq!(m.as_slice(), &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
    let d = Matrix::from_diagonal(&[1.0, 2.0, 3.0]);
    assert_eq!(d[(1, 1)], 2.0);
    assert_eq!(d[(1, 2)], 0.0);
}

#[test]
fn get_and_set_check_bounds() {
    let mut m = Matrix::zeros(2, 2);
    assert_eq!(m.get(1, 1), Some(0.0));
    assert_eq!(m.get(2, 0), None);
    m.set(0, 1, 7.0).unwrap();
    assert_eq!(m.try_get(0, 1), Ok(7.0));
    assert_eq!(
        m.set(0, 2, 1.0),
        Err(LinalgError::IndexOutOfBounds {
            index: (0, 2),
            shape: (2, 2)
        })
    );
    assert_eq!(
        m.try_get(5, 0),
        Err(LinalgError::IndexOutOfBounds {
            index: (5, 0),
            shape: (2, 2)
        })
    );
}

#[test]
fn index_mut_writes_through() {
    let mut m = Matrix::zeros(2, 2);
    m[(1, 0)] = -3.0;
    assert_eq!(m.as_slice(), &[0.0, 0.0, -3.0, 0.0]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn indexing_past_the_last_column_panics() {
    let m = Matrix::zeros(2, 2);
    // Offset 1 * 2 + 2 would still be inside the buffer of a larger matrix, so the column
    // has to be checked on its own.
    let _ = m[(0, 2)];
}
