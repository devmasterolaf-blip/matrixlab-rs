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

mod csr {
    use super::{coo, dense};
    use matrixlab::sparse::{CooMatrix, CsrMatrix};
    use matrixlab::{LinalgError, Matrix, Vector};

    #[test]
    fn conversion_sorts_rows_and_columns() {
        let csr = coo().to_csr();
        assert_eq!(csr.row_offsets(), &[0, 2, 3, 5]);
        assert_eq!(csr.col_indices(), &[0, 2, 1, 0, 2]);
        assert_eq!(csr.values(), &[4.0, 1.0, 3.0, 2.0, 5.0]);
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.to_dense(), dense());
    }

    #[test]
    fn conversion_merges_duplicates() {
        let coo = CooMatrix::from_triplets(
            2,
            3,
            &[(1, 2, 1.0), (0, 1, 2.0), (1, 2, 3.0), (1, 0, 5.0), (1, 2, -0.5)],
        )
        .unwrap();
        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.row(1), (&[0usize, 2][..], &[5.0, 3.5][..]));
        assert_eq!(csr.to_dense(), coo.to_dense());
    }

    #[test]
    fn empty_rows_are_represented_by_equal_offsets() {
        let coo = CooMatrix::from_triplets(4, 2, &[(0, 0, 1.0), (3, 1, 2.0)]).unwrap();
        let csr = coo.to_csr();
        assert_eq!(csr.row_offsets(), &[0, 1, 1, 1, 2]);
        assert_eq!(csr.row(1).0.len(), 0);
        assert_eq!(csr.get(2, 1), Some(0.0));
    }

    #[test]
    fn a_matrix_without_entries_converts_cleanly() {
        let csr = CooMatrix::new(3, 3).to_csr();
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.row_offsets(), &[0, 0, 0, 0]);
        assert_eq!(csr.to_dense(), Matrix::zeros(3, 3));
        assert_eq!(
            csr.mul_vec(&Vector::filled(3, 1.0)).unwrap(),
            Vector::zeros(3)
        );
    }

    #[test]
    fn get_distinguishes_missing_from_out_of_bounds() {
        let csr = coo().to_csr();
        assert_eq!(csr.get(0, 2), Some(1.0));
        assert_eq!(csr.get(0, 1), Some(0.0));
        assert_eq!(csr.get(3, 0), None);
        assert_eq!(csr.get(0, 3), None);
    }

    #[test]
    fn diagonal_includes_structural_zeros() {
        let csr = CooMatrix::from_triplets(3, 3, &[(0, 0, 2.0), (2, 2, 7.0), (1, 0, 1.0)])
            .unwrap()
            .to_csr();
        assert_eq!(csr.diagonal(), vec![2.0, 0.0, 7.0]);
    }

    #[test]
    fn matrix_vector_product_matches_the_dense_one() {
        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let csr = coo().to_csr();
        assert_eq!(csr.mul_vec(&x).unwrap().as_slice(), &[7.0, 6.0, 17.0]);
        assert_eq!(
            csr.mul_vec(&Vector::zeros(4)),
            Err(LinalgError::DimensionMismatch {
                op: "sparse matrix-vector product",
                left: (3, 3),
                right: (4, 1)
            })
        );
    }

    #[test]
    fn rectangular_product_and_transpose() {
        let coo = CooMatrix::from_triplets(2, 3, &[(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0)])
            .unwrap();
        let csr = coo.to_csr();
        let y = csr.mul_vec(&Vector::from_slice(&[1.0, 1.0, 1.0])).unwrap();
        assert_eq!(y.as_slice(), &[3.0, 3.0]);
        let t = csr.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.to_dense(), csr.to_dense().transpose());
        assert_eq!(t.transpose(), csr);
    }

    #[test]
    fn round_trip_through_coo_and_dense() {
        let csr = CsrMatrix::from_dense(&dense());
        assert_eq!(csr.to_coo().to_csr(), csr);
        assert_eq!(CsrMatrix::from_dense(&csr.to_dense()), csr);
        let triplets: Vec<_> = csr.iter().collect();
        assert_eq!(triplets[0], (0, 0, 4.0));
        assert_eq!(triplets[4], (2, 2, 5.0));
    }

    #[test]
    fn identity_behaves_like_the_identity() {
        let i = CsrMatrix::identity(4);
        let x = Vector::from_slice(&[1.0, -2.0, 3.0, -4.0]);
        assert_eq!(i.mul_vec(&x).unwrap(), x);
        assert_eq!(i.to_dense(), Matrix::identity(4));
    }

    #[test]
    fn symmetry_check_looks_at_structure_and_values() {
        assert!(!coo().to_csr().is_symmetric(1e-12));
        let sym = CooMatrix::from_triplets(
            3,
            3,
            &[(0, 0, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0), (2, 2, 1.0)],
        )
        .unwrap()
        .to_csr();
        assert!(sym.is_symmetric(0.0));
        let one_sided = CooMatrix::from_triplets(2, 2, &[(0, 1, 1.0)]).unwrap().to_csr();
        assert!(!one_sided.is_symmetric(0.5));
        assert!(!CooMatrix::new(2, 3).to_csr().is_symmetric(1.0));
    }

    #[test]
    fn raw_parts_are_validated() {
        let ok = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]);
        assert_eq!(ok.unwrap().to_dense(), Matrix::from_diagonal(&[1.0, 2.0]));

        let reason = |result: matrixlab::Result<CsrMatrix>| match result {
            Err(LinalgError::InvalidStructure { reason }) => reason,
            other => panic!("expected InvalidStructure, got {:?}", other),
        };
        assert_eq!(
            reason(CsrMatrix::from_raw_parts(2, 2, vec![0, 1], vec![0], vec![1.0])),
            "row_offsets must have rows + 1 elements"
        );
        assert_eq!(
            reason(CsrMatrix::from_raw_parts(1, 2, vec![0, 2], vec![0], vec![1.0, 2.0])),
            "col_indices and values must have the same length"
        );
        assert_eq!(
            reason(CsrMatrix::from_raw_parts(1, 2, vec![1, 1], vec![0], vec![1.0])),
            "row_offsets must start at zero"
        );
        assert_eq!(
            reason(CsrMatrix::from_raw_parts(2, 2, vec![0, 2, 1], vec![0, 1], vec![1.0, 2.0])),
            "row_offsets must end at the number of entries"
        );
        assert_eq!(
            reason(CsrMatrix::from_raw_parts(
                3,
                2,
                vec![0, 2, 1, 2],
                vec![0, 1],
                vec![1.0, 2.0]
            )),
            "row_offsets must not decrease"
        );
        assert_eq!(
            reason(CsrMatrix::from_raw_parts(1, 2, vec![0, 1], vec![2], vec![1.0])),
            "a column index is out of range"
        );
        assert_eq!(
            reason(CsrMatrix::from_raw_parts(1, 3, vec![0, 2], vec![1, 1], vec![1.0, 2.0])),
            "column indices must be strictly increasing within a row"
        );
    }
}
