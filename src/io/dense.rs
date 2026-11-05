//! The plain-text format for dense matrices and vectors.
//!
//! One matrix row per line. Fields are separated by commas or by blanks: a line that
//! contains a comma is split at its commas, any other line at runs of whitespace. `#` and
//! `%` start a comment that runs to the end of the line; blank lines are skipped.
//!
//! ```text
//! # a 2 x 3 matrix
//! 1.0  2.0  3.0
//! 4.0, 5.0, 6.0   % commas work too
//! ```

use crate::matrix::Matrix;
use crate::vector::Vector;

use super::error::{ParseError, ParseErrorKind};

/// The field separator used when writing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
    /// A single space between fields.
    Space,
    /// A comma between fields, which makes the output a CSV file.
    Comma,
}

/// Parses a dense matrix.
pub fn parse_matrix(text: &str) -> Result<Matrix, ParseError> {
    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;
    let mut last_line = 0;
    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        last_line = line_number;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let before = data.len();
        for (position, field) in split_fields(line).enumerate() {
            let value = field.parse::<f64>().map_err(|_| {
                ParseError::new(
                    line_number,
                    ParseErrorKind::InvalidNumber {
                        column: position + 1,
                        token: field.to_string(),
                    },
                )
            })?;
            data.push(value);
        }
        let found = data.len() - before;
        if rows == 0 {
            cols = found;
        } else if found != cols {
            return Err(ParseError::new(
                line_number,
                ParseErrorKind::RaggedRow {
                    expected: cols,
                    found,
                },
            ));
        }
        rows += 1;
    }
    if rows == 0 {
        return Err(ParseError::new(last_line, ParseErrorKind::EmptyInput));
    }
    Ok(Matrix::from_parts(rows, cols, data))
}

/// Parses a vector: either one value per line or all values on one line.
pub fn parse_vector(text: &str) -> Result<Vector, ParseError> {
    let matrix = parse_matrix(text)?;
    if matrix.rows() != 1 && matrix.cols() != 1 {
        return Err(ParseError::new(
            text.lines().count(),
            ParseErrorKind::NotAVector {
                rows: matrix.rows(),
                cols: matrix.cols(),
            },
        ));
    }
    Ok(Vector::new(matrix.into_vec()))
}

/// Writes a matrix, one row per line, each line ending in a newline.
///
/// Numbers are written with the shortest decimal text that reads back as the same `f64`,
/// so `parse_matrix(&write_matrix(&m, d))` gives `m` again for finite values.
pub fn write_matrix(matrix: &Matrix, delimiter: Delimiter) -> String {
    let separator = match delimiter {
        Delimiter::Space => " ",
        Delimiter::Comma => ",",
    };
    let mut out = String::new();
    for row in matrix.iter_rows() {
        let fields: Vec<String> = row.iter().map(|x| x.to_string()).collect();
        out.push_str(&fields.join(separator));
        out.push('\n');
    }
    out
}

/// Writes a vector, one value per line.
pub fn write_vector(vector: &Vector) -> String {
    let mut out = String::new();
    for x in vector.iter() {
        out.push_str(&x.to_string());
        out.push('\n');
    }
    out
}

/// The part of a line before the first `#` or `%`.
fn strip_comment(line: &str) -> &str {
    match line.find(|c| c == '#' || c == '%') {
        Some(position) => &line[..position],
        None => line,
    }
}

/// Splits a non-empty, trimmed line into fields.
fn split_fields(line: &str) -> Box<dyn Iterator<Item = &str> + '_> {
    if line.contains(',') {
        Box::new(line.split(',').map(str::trim))
    } else {
        Box::new(line.split_whitespace())
    }
}
