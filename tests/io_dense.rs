use matrixlab::io::dense::{parse_matrix, parse_vector, write_matrix, write_vector, Delimiter};
use matrixlab::io::{ParseError, ParseErrorKind};
use matrixlab::{Matrix, Vector};

#[test]
fn whitespace_separated_rows() {
    let m = parse_matrix("1 2 3\n4 5 6\n").unwrap();
    assert_eq!(m, Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap());
}

#[test]
fn comma_separated_rows_with_optional_blanks() {
    let m = parse_matrix("1,2.5\n-3 , 4e2\n").unwrap();
    assert_eq!(m, Matrix::from_rows(&[[1.0, 2.5], [-3.0, 400.0]]).unwrap());
}

#[test]
fn comments_blank_lines_tabs_and_windows_line_endings_are_tolerated() {
    let text = "# header\r\n\r\n1\t2   % trailing comment\r\n   \r\n3 4\r\n% footer";
    let m = parse_matrix(text).unwrap();
    assert_eq!(m, Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap());
}

#[test]
fn a_missing_final_newline_is_fine() {
    assert_eq!(parse_matrix("7").unwrap(), Matrix::from_rows(&[[7.0]]).unwrap());
}

#[test]
fn an_invalid_number_is_located_by_line_and_field() {
    let err = parse_matrix("1 2\n3 x4\n").unwrap_err();
    assert_eq!(
        err,
        ParseError {
            line: 2,
            kind: ParseErrorKind::InvalidNumber {
                column: 2,
                token: "x4".to_string()
            }
        }
    );
    assert_eq!(err.to_string(), "line 2: field 2 is not a number: 'x4'");
}

#[test]
fn an_empty_csv_field_is_an_invalid_number() {
    let err = parse_matrix("1,,3\n").unwrap_err();
    assert_eq!(
        err.kind,
        ParseErrorKind::InvalidNumber {
            column: 2,
            token: String::new()
        }
    );
}

#[test]
fn a_ragged_row_is_reported_with_both_counts() {
    let err = parse_matrix("1 2 3\n# comment\n4 5\n").unwrap_err();
    assert_eq!(
        err,
        ParseError {
            line: 3,
            kind: ParseErrorKind::RaggedRow {
                expected: 3,
                found: 2
            }
        }
    );
    assert_eq!(err.to_string(), "line 3: expected 3 fields, found 2");
}

#[test]
fn input_without_data_is_an_error() {
    assert_eq!(parse_matrix("").unwrap_err().kind, ParseErrorKind::EmptyInput);
    let err = parse_matrix("# only\n\n% comments\n").unwrap_err();
    assert_eq!(err.kind, ParseErrorKind::EmptyInput);
    assert_eq!(err.line, 3);
}

#[test]
fn vectors_can_be_a_column_or_a_row() {
    assert_eq!(
        parse_vector("1\n2\n3\n").unwrap(),
        Vector::from_slice(&[1.0, 2.0, 3.0])
    );
    assert_eq!(
        parse_vector("1, 2, 3").unwrap(),
        Vector::from_slice(&[1.0, 2.0, 3.0])
    );
}

#[test]
fn a_matrix_is_not_a_vector() {
    let err = parse_vector("1 2\n3 4\n").unwrap_err();
    assert_eq!(err.kind, ParseErrorKind::NotAVector { rows: 2, cols: 2 });
    assert_eq!(
        err.to_string(),
        "line 2: expected a single row or a single column, found a 2x2 matrix"
    );
}

#[test]
fn writing_uses_the_chosen_delimiter() {
    let m = Matrix::from_rows(&[[1.0, -2.5], [1e-7, 4.0]]).unwrap();
    assert_eq!(write_matrix(&m, Delimiter::Space), "1 -2.5\n0.0000001 4\n");
    assert_eq!(write_matrix(&m, Delimiter::Comma), "1,-2.5\n0.0000001,4\n");
    assert_eq!(
        write_vector(&Vector::from_slice(&[0.5, 2.0])),
        "0.5\n2\n"
    );
}

#[test]
fn written_text_reads_back_bit_for_bit() {
    let m = Matrix::from_rows(&[
        [1.0 / 3.0, std::f64::consts::PI, -1e-300],
        [6.02214076e23, f64::MIN_POSITIVE, 0.1 + 0.2],
    ])
    .unwrap();
    for delimiter in [Delimiter::Space, Delimiter::Comma] {
        let back = parse_matrix(&write_matrix(&m, delimiter)).unwrap();
        assert_eq!(back, m);
    }
    let v = Vector::from_slice(&[1.0 / 7.0, -2.0e-9]);
    assert_eq!(parse_vector(&write_vector(&v)).unwrap(), v);
}
