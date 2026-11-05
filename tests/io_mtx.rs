use matrixlab::io::mtx::{is_matrix_market, parse_coordinate, write_coordinate};
use matrixlab::io::{ParseError, ParseErrorKind};
use matrixlab::sparse::CooMatrix;
use matrixlab::Matrix;

const GENERAL: &str = "%%MatrixMarket matrix coordinate real general
% a comment

3 4 4
1 1 2.0
2 2 3.5
3 1 -1.0
3 4 4e1
";

fn kind_of(text: &str) -> ParseErrorKind {
    parse_coordinate(text).unwrap_err().kind
}

#[test]
fn reads_a_general_real_matrix() {
    let coo = parse_coordinate(GENERAL).unwrap();
    assert_eq!(coo.shape(), (3, 4));
    assert_eq!(coo.nnz(), 4);
    let expected = Matrix::from_rows(&[
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.5, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 40.0],
    ])
    .unwrap();
    assert_eq!(coo.to_dense(), expected);
}

#[test]
fn a_symmetric_file_fills_in_the_other_triangle() {
    let text = "%%MatrixMarket matrix coordinate real symmetric
3 3 4
1 1 4
2 1 1
3 2 -2
3 3 5
";
    let coo = parse_coordinate(text).unwrap();
    assert_eq!(coo.nnz(), 6);
    let expected =
        Matrix::from_rows(&[[4.0, 1.0, 0.0], [1.0, 0.0, -2.0], [0.0, -2.0, 5.0]]).unwrap();
    assert_eq!(coo.to_dense(), expected);
}

#[test]
fn a_skew_symmetric_file_mirrors_with_the_opposite_sign() {
    let text = "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 1 3\n";
    let coo = parse_coordinate(text).unwrap();
    let expected = Matrix::from_rows(&[[0.0, -3.0], [3.0, 0.0]]).unwrap();
    assert_eq!(coo.to_dense(), expected);
}

#[test]
fn a_skew_symmetric_diagonal_entry_is_rejected() {
    let text = "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 2 3\n";
    let err = parse_coordinate(text).unwrap_err();
    assert_eq!(err.line, 3);
    assert_eq!(err.kind, ParseErrorKind::SkewDiagonalEntry { index: 2 });
}

#[test]
fn pattern_entries_are_ones() {
    let text = "%%MatrixMarket matrix coordinate pattern general\n2 3 2\n1 3\n2 1\n";
    let coo = parse_coordinate(text).unwrap();
    let expected = Matrix::from_rows(&[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).unwrap();
    assert_eq!(coo.to_dense(), expected);
}

#[test]
fn integer_fields_must_hold_whole_numbers() {
    let good = "%%MatrixMarket matrix coordinate integer general\n1 1 1\n1 1 -7\n";
    assert_eq!(parse_coordinate(good).unwrap().to_dense()[(0, 0)], -7.0);

    let bad = "%%MatrixMarket matrix coordinate integer general\n1 1 1\n1 1 2.5\n";
    assert_eq!(
        kind_of(bad),
        ParseErrorKind::InvalidNumber {
            column: 3,
            token: "2.5".to_string()
        }
    );
}

#[test]
fn banner_words_are_case_insensitive() {
    let text = "%%MatrixMarket MATRIX Coordinate Real GENERAL\n1 1 1\n1 1 2\n";
    assert_eq!(parse_coordinate(text).unwrap().nnz(), 1);
}

#[test]
fn repeated_entries_add_up_in_the_dense_form() {
    let text = "%%MatrixMarket matrix coordinate real general\n1 1 2\n1 1 1.5\n1 1 2.5\n";
    assert_eq!(parse_coordinate(text).unwrap().to_dense()[(0, 0)], 4.0);
}

#[test]
fn a_matrix_without_entries_is_valid() {
    let coo = parse_coordinate("%%MatrixMarket matrix coordinate real general\n2 5 0\n").unwrap();
    assert_eq!(coo.shape(), (2, 5));
    assert_eq!(coo.nnz(), 0);
}

#[test]
fn the_banner_is_required() {
    assert_eq!(kind_of("3 3 1\n1 1 1\n"), ParseErrorKind::MissingBanner);
    assert_eq!(kind_of(""), ParseErrorKind::EmptyInput);
    assert_eq!(
        kind_of("%%MatrixMarket matrix coordinate real\n"),
        ParseErrorKind::MalformedBanner { found: 4 }
    );
}

#[test]
fn unsupported_variants_name_the_offending_word() {
    let cases = [
        ("vector coordinate real general", "object", "vector"),
        ("matrix array real general", "format", "array"),
        ("matrix coordinate complex general", "field", "complex"),
        ("matrix coordinate real hermitian", "symmetry", "hermitian"),
        ("matrix coordinate pattern skew-symmetric", "symmetry", "skew-symmetric"),
    ];
    for (words, what, token) in cases {
        let text = format!("%%MatrixMarket {}\n1 1 0\n", words);
        let err = parse_coordinate(&text).unwrap_err();
        assert_eq!(err.line, 1);
        assert_eq!(
            err.kind,
            ParseErrorKind::Unsupported {
                what,
                token: token.to_string()
            }
        );
    }
    let err = parse_coordinate("%%MatrixMarket matrix array real general\n").unwrap_err();
    assert_eq!(err.to_string(), "line 1: unsupported format 'array'");
}

#[test]
fn the_size_line_is_checked() {
    let banner = "%%MatrixMarket matrix coordinate real general\n";
    assert_eq!(kind_of(banner), ParseErrorKind::EmptyInput);
    assert_eq!(
        kind_of(&format!("{}3 3\n", banner)),
        ParseErrorKind::FieldCount {
            expected: 3,
            found: 2
        }
    );
    assert_eq!(
        kind_of(&format!("{}3 -3 1\n", banner)),
        ParseErrorKind::InvalidInteger {
            column: 2,
            token: "-3".to_string()
        }
    );
}

#[test]
fn a_symmetric_matrix_must_be_square() {
    let text = "%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n";
    assert_eq!(
        kind_of(text),
        ParseErrorKind::SymmetryNeedsSquare { rows: 2, cols: 3 }
    );
}

#[test]
fn entries_are_checked_field_by_field() {
    let banner = "%%MatrixMarket matrix coordinate real general\n2 2 1\n";
    assert_eq!(
        parse_coordinate(&format!("{}1 1\n", banner)).unwrap_err(),
        ParseError {
            line: 3,
            kind: ParseErrorKind::FieldCount {
                expected: 3,
                found: 2
            }
        }
    );
    assert_eq!(
        kind_of(&format!("{}1 one 2.0\n", banner)),
        ParseErrorKind::InvalidInteger {
            column: 2,
            token: "one".to_string()
        }
    );
    assert_eq!(
        kind_of(&format!("{}1 1 2.0.0\n", banner)),
        ParseErrorKind::InvalidNumber {
            column: 3,
            token: "2.0.0".to_string()
        }
    );
}

#[test]
fn indices_are_one_based_and_bounded() {
    let banner = "%%MatrixMarket matrix coordinate real general\n2 3 1\n";
    assert_eq!(
        kind_of(&format!("{}0 1 1.0\n", banner)),
        ParseErrorKind::IndexOutOfRange { index: 0, limit: 2 }
    );
    let err = parse_coordinate(&format!("{}2 4 1.0\n", banner)).unwrap_err();
    assert_eq!(err.kind, ParseErrorKind::IndexOutOfRange { index: 4, limit: 3 });
    assert_eq!(err.to_string(), "line 3: index 4 is outside 1..=3");
}

#[test]
fn the_entry_count_must_match_the_size_line() {
    let short = "%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 1\n2 2 1\n";
    let err = parse_coordinate(short).unwrap_err();
    assert_eq!(err.line, 4);
    assert_eq!(
        err.kind,
        ParseErrorKind::EntryCount {
            expected: 3,
            found: 2
        }
    );

    let long = "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 1\n2 2 1\n";
    let err = parse_coordinate(long).unwrap_err();
    assert_eq!(err.line, 4);
    assert_eq!(
        err.kind,
        ParseErrorKind::EntryCount {
            expected: 1,
            found: 2
        }
    );
}

#[test]
fn writing_produces_a_general_real_file() {
    let coo = CooMatrix::from_triplets(2, 3, &[(0, 0, 1.5), (1, 2, -2.0)]).unwrap();
    assert_eq!(
        write_coordinate(&coo),
        "%%MatrixMarket matrix coordinate real general\n2 3 2\n1 1 1.5\n2 3 -2\n"
    );
}

#[test]
fn written_files_read_back_unchanged() {
    let coo = CooMatrix::from_triplets(
        3,
        3,
        &[(2, 0, 1.0 / 3.0), (0, 1, -1e-200), (1, 1, 6.02e23), (0, 1, 0.5)],
    )
    .unwrap();
    assert_eq!(parse_coordinate(&write_coordinate(&coo)).unwrap(), coo);
}

#[test]
fn the_banner_identifies_the_format() {
    assert!(is_matrix_market(GENERAL));
    assert!(!is_matrix_market("1 2\n3 4\n"));
}
