mod common;

use std::io::ErrorKind;
use std::path::Path;

use common::scratch_file;
use matrixlab::io::{
    parse_matrix_auto, read_matrix, read_vector, MatrixData, ParseErrorKind, ReadError,
};
use matrixlab::{Matrix, Vector};

const SPARSE: &str = "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 4\n2 2 5\n";

#[test]
fn the_format_is_detected_from_the_content() {
    let dense = parse_matrix_auto("4 0\n0 5\n").unwrap();
    let sparse = parse_matrix_auto(SPARSE).unwrap();
    assert!(!dense.is_sparse());
    assert!(sparse.is_sparse());
    assert_eq!(dense.shape(), (2, 2));
    assert_eq!(sparse.shape(), (2, 2));
    assert_eq!(dense.to_dense(), sparse.to_dense());
    assert_eq!(dense.to_csr(), sparse.to_csr());
}

#[test]
fn a_dense_file_is_read_as_a_dense_matrix() {
    let path = scratch_file("io_file_dense.txt", "1 2\n3 4\n");
    let expected = Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(read_matrix(&path).unwrap(), MatrixData::Dense(expected));
}

#[test]
fn a_matrix_market_file_is_read_as_a_sparse_matrix() {
    let path = scratch_file("io_file_sparse.mtx", SPARSE);
    let data = read_matrix(&path).unwrap();
    assert!(data.is_sparse());
    assert_eq!(data.to_dense(), Matrix::from_diagonal(&[4.0, 5.0]));
}

#[test]
fn a_vector_file_is_read() {
    let path = scratch_file("io_file_vector.txt", "# b\n1.5\n-2\n");
    assert_eq!(read_vector(&path).unwrap(), Vector::from_slice(&[1.5, -2.0]));
}

#[test]
fn a_missing_file_is_an_io_error_that_names_the_path() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("matrixlab-tests-no-such-file.txt");
    match read_matrix(&path).unwrap_err() {
        ReadError::Io {
            path: reported,
            kind,
            ..
        } => {
            assert_eq!(reported, path);
            assert_eq!(kind, ErrorKind::NotFound);
        }
        other => panic!("expected an I/O error, got {:?}", other),
    }
    let message = read_vector(&path).unwrap_err().to_string();
    assert!(message.starts_with("cannot read "), "{}", message);
    assert!(message.contains("matrixlab-tests-no-such-file.txt"), "{}", message);
}

#[test]
fn a_parse_error_keeps_the_path_and_the_line() {
    let path = scratch_file("io_file_bad.txt", "1 2\n3 oops\n");
    let err = read_matrix(&path).unwrap_err();
    match &err {
        ReadError::Parse {
            path: reported,
            error,
        } => {
            assert_eq!(reported, &path);
            assert_eq!(error.line, 2);
            assert_eq!(
                error.kind,
                ParseErrorKind::InvalidNumber {
                    column: 2,
                    token: "oops".to_string()
                }
            );
        }
        other => panic!("expected a parse error, got {:?}", other),
    }
    assert!(err
        .to_string()
        .ends_with("io_file_bad.txt: line 2: field 2 is not a number: 'oops'"));
}

#[test]
fn a_matrix_market_file_is_not_a_vector_file() {
    let path = scratch_file("io_file_vector.mtx", SPARSE);
    assert!(matches!(
        read_vector(&path).unwrap_err(),
        ReadError::Parse { .. }
    ));
}
