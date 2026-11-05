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

fn sample() -> Matrix {
    Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap()
}

#[test]
fn rows_and_columns_can_be_read() {
    let m = sample();
    assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
    assert_eq!(m.column(2), vec![3.0, 6.0]);
    let rows: Vec<&[f64]> = m.iter_rows().collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], &[1.0, 2.0, 3.0]);
}

#[test]
fn row_mut_edits_in_place() {
    let mut m = sample();
    m.row_mut(0)[1] = 20.0;
    assert_eq!(m[(0, 1)], 20.0);
}

#[test]
fn transpose_swaps_the_shape_and_is_an_involution() {
    let m = sample();
    let t = m.transpose();
    assert_eq!(t.shape(), (3, 2));
    assert_eq!(t.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(t.transpose(), m);
}

#[test]
fn diagonal_of_a_rectangular_matrix_stops_at_the_short_side() {
    assert_eq!(sample().diagonal(), vec![1.0, 5.0]);
    assert_eq!(sample().transpose().diagonal(), vec![1.0, 5.0]);
}

#[test]
fn trace_needs_a_square_matrix() {
    let sq = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(sq.trace(), Ok(5.0));
    assert_eq!(
        sample().trace(),
        Err(LinalgError::NotSquare { rows: 2, cols: 3 })
    );
}

#[test]
fn map_applies_the_closure_to_every_element() {
    let doubled = sample().map(|x| 2.0 * x);
    assert_eq!(doubled.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn swap_rows_works_in_both_orders_and_for_equal_indices() {
    let mut m = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).unwrap();
    m.swap_rows(0, 2);
    assert_eq!(m.as_slice(), &[5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    m.swap_rows(2, 1);
    assert_eq!(m.as_slice(), &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    m.swap_rows(1, 1);
    assert_eq!(m.as_slice(), &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn symmetry_uses_an_absolute_tolerance() {
    let s = Matrix::from_rows(&[[2.0, 1.0], [1.0 + 1e-12, 3.0]]).unwrap();
    assert!(s.is_symmetric(1e-10));
    assert!(!s.is_symmetric(1e-14));
    assert!(!sample().is_symmetric(1e-10));
    let with_nan = Matrix::from_rows(&[[1.0, f64::NAN], [0.0, 1.0]]).unwrap();
    assert!(!with_nan.is_symmetric(1.0));
}

#[test]
fn from_vec_rejects_a_shape_whose_element_count_overflows() {
    // Half the address space times two wraps around to zero, which must not pass for an
    // empty buffer.
    let half = usize::MAX / 2 + 1;
    let err = Matrix::from_vec(half, 2, Vec::new()).unwrap_err();
    assert_eq!(
        err,
        LinalgError::InvalidShape {
            rows: half,
            cols: 2,
            len: 0
        }
    );
    assert_eq!(
        err.to_string(),
        format!("a {}x2 matrix has more elements than can be addressed", half)
    );
    assert!(Matrix::from_vec(usize::MAX, usize::MAX, vec![1.0]).is_err());
}

#[test]
fn trace_of_an_empty_matrix_is_positive_zero() {
    let trace = Matrix::zeros(0, 0).trace().unwrap();
    assert_eq!(trace, 0.0);
    assert!(trace.is_sign_positive());
}

#[test]
fn trace_survives_cancellation_between_large_diagonal_elements() {
    // A plain running sum loses the 1 next to 1e16.
    let m = Matrix::from_diagonal(&[1e16, 1.0, -1e16]);
    assert_eq!(m.trace(), Ok(1.0));
}

#[test]
fn swap_columns_exchanges_every_row() {
    let mut m = Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    m.swap_columns(0, 2);
    assert_eq!(m.as_slice(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    m.swap_columns(1, 1);
    assert_eq!(m.as_slice(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
}

#[test]
#[should_panic(expected = "cannot swap columns 0 and 3 of a matrix with 3 columns")]
fn swap_columns_panics_out_of_bounds() {
    Matrix::zeros(2, 3).swap_columns(0, 3);
}
