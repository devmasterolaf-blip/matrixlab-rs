//! The Matrix Market coordinate format.
//!
//! ```text
//! %%MatrixMarket matrix coordinate real general
//! % comment lines start with a percent sign
//! 3 3 4
//! 1 1 2.0
//! 2 2 3.0
//! 3 1 -1.0
//! 3 3 4.0
//! ```
//!
//! The reader accepts the fields `real`, `integer` and `pattern` and the symmetries
//! `general`, `symmetric` and `skew-symmetric`. The dense `array` format, `complex` values
//! and `hermitian` symmetry are reported as [`ParseErrorKind::Unsupported`]. Indices in the
//! file are 1-based; the matrices in memory are 0-based.

use crate::sparse::CooMatrix;

use super::error::{ParseError, ParseErrorKind};

/// The first word of every Matrix Market file.
pub const BANNER: &str = "%%MatrixMarket";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    Real,
    Integer,
    Pattern,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symmetry {
    General,
    Symmetric,
    SkewSymmetric,
}

/// True when `text` starts with the Matrix Market banner word.
pub fn is_matrix_market(text: &str) -> bool {
    text.starts_with(BANNER)
}

/// Parses a coordinate-format file into a [`CooMatrix`].
///
/// A symmetric file stores one triangle; the other one is filled in, so the result is the
/// full matrix. A `pattern` file gives every listed position the value 1.
pub fn parse_coordinate(text: &str) -> Result<CooMatrix, ParseError> {
    let mut lines = text.lines().enumerate().map(|(index, line)| (index + 1, line));

    let (field, symmetry) = match lines.next() {
        Some((_, banner)) => parse_banner(banner)?,
        None => return Err(ParseError::new(0, ParseErrorKind::EmptyInput)),
    };

    let mut last_line = 1;
    let mut size = None;
    for (number, raw) in lines.by_ref() {
        last_line = number;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        size = Some(parse_size(number, line)?);
        break;
    }
    let (rows, cols, announced) = match size {
        Some(size) => size,
        None => return Err(ParseError::new(last_line, ParseErrorKind::EmptyInput)),
    };
    if symmetry != Symmetry::General && rows != cols {
        return Err(ParseError::new(
            last_line,
            ParseErrorKind::SymmetryNeedsSquare { rows, cols },
        ));
    }

    let expected_fields = if field == Field::Pattern { 2 } else { 3 };
    let mut coo = CooMatrix::new(rows, cols);
    let mut found = 0;
    for (number, raw) in lines {
        last_line = number;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        found += 1;
        if found > announced {
            return Err(ParseError::new(
                number,
                ParseErrorKind::EntryCount {
                    expected: announced,
                    found,
                },
            ));
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() != expected_fields {
            return Err(ParseError::new(
                number,
                ParseErrorKind::FieldCount {
                    expected: expected_fields,
                    found: tokens.len(),
                },
            ));
        }
        let row = parse_index(number, 1, tokens[0], rows)?;
        let col = parse_index(number, 2, tokens[1], cols)?;
        let value = match field {
            Field::Pattern => 1.0,
            Field::Real => parse_real(number, tokens[2])?,
            Field::Integer => parse_whole(number, tokens[2])?,
        };
        push(&mut coo, row, col, value);
        if row != col {
            match symmetry {
                Symmetry::General => {}
                Symmetry::Symmetric => push(&mut coo, col, row, value),
                Symmetry::SkewSymmetric => push(&mut coo, col, row, -value),
            }
        } else if symmetry == Symmetry::SkewSymmetric {
            return Err(ParseError::new(
                number,
                ParseErrorKind::SkewDiagonalEntry { index: row + 1 },
            ));
        }
    }
    if found != announced {
        return Err(ParseError::new(
            last_line,
            ParseErrorKind::EntryCount {
                expected: announced,
                found,
            },
        ));
    }
    Ok(coo)
}

/// Writes a matrix as `matrix coordinate real general`, entries in stored order.
///
/// Values use the shortest text that reads back as the same `f64`.
pub fn write_coordinate(matrix: &CooMatrix) -> String {
    let mut out = format!("{} matrix coordinate real general\n", BANNER);
    out.push_str(&format!(
        "{} {} {}\n",
        matrix.rows(),
        matrix.cols(),
        matrix.nnz()
    ));
    for (row, col, value) in matrix.iter() {
        out.push_str(&format!("{} {} {}\n", row + 1, col + 1, value));
    }
    out
}

fn parse_banner(line: &str) -> Result<(Field, Symmetry), ParseError> {
    let error = |kind| ParseError::new(1, kind);
    let words: Vec<&str> = line.split_whitespace().collect();
    if words.first() != Some(&BANNER) {
        return Err(error(ParseErrorKind::MissingBanner));
    }
    if words.len() != 5 {
        return Err(error(ParseErrorKind::MalformedBanner { found: words.len() }));
    }
    let unsupported = |what, token: &str| {
        error(ParseErrorKind::Unsupported {
            what,
            token: token.to_string(),
        })
    };
    // The words after the banner are case-insensitive.
    let lower: Vec<String> = words[1..].iter().map(|w| w.to_ascii_lowercase()).collect();
    if lower[0] != "matrix" {
        return Err(unsupported("object", words[1]));
    }
    if lower[1] != "coordinate" {
        return Err(unsupported("format", words[2]));
    }
    let field = match lower[2].as_str() {
        "real" => Field::Real,
        "integer" => Field::Integer,
        "pattern" => Field::Pattern,
        _ => return Err(unsupported("field", words[3])),
    };
    let symmetry = match lower[3].as_str() {
        "general" => Symmetry::General,
        "symmetric" => Symmetry::Symmetric,
        "skew-symmetric" => Symmetry::SkewSymmetric,
        _ => return Err(unsupported("symmetry", words[4])),
    };
    if field == Field::Pattern && symmetry == Symmetry::SkewSymmetric {
        // A pattern has no signs to flip.
        return Err(unsupported("symmetry", words[4]));
    }
    Ok((field, symmetry))
}

fn parse_size(line_number: usize, line: &str) -> Result<(usize, usize, usize), ParseError> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() != 3 {
        return Err(ParseError::new(
            line_number,
            ParseErrorKind::FieldCount {
                expected: 3,
                found: tokens.len(),
            },
        ));
    }
    let mut numbers = [0usize; 3];
    for (position, token) in tokens.iter().enumerate() {
        numbers[position] = parse_unsigned(line_number, position + 1, token)?;
    }
    Ok((numbers[0], numbers[1], numbers[2]))
}

fn parse_unsigned(line: usize, column: usize, token: &str) -> Result<usize, ParseError> {
    token.parse::<usize>().map_err(|_| {
        ParseError::new(
            line,
            ParseErrorKind::InvalidInteger {
                column,
                token: token.to_string(),
            },
        )
    })
}

/// Parses a 1-based index and returns it 0-based.
fn parse_index(line: usize, column: usize, token: &str, limit: usize) -> Result<usize, ParseError> {
    let index = parse_unsigned(line, column, token)?;
    if index == 0 || index > limit {
        return Err(ParseError::new(
            line,
            ParseErrorKind::IndexOutOfRange { index, limit },
        ));
    }
    Ok(index - 1)
}

fn parse_real(line: usize, token: &str) -> Result<f64, ParseError> {
    token.parse::<f64>().map_err(|_| invalid_value(line, token))
}

fn parse_whole(line: usize, token: &str) -> Result<f64, ParseError> {
    token
        .parse::<i64>()
        .map(|value| value as f64)
        .map_err(|_| invalid_value(line, token))
}

fn invalid_value(line: usize, token: &str) -> ParseError {
    ParseError::new(
        line,
        ParseErrorKind::InvalidNumber {
            column: 3,
            token: token.to_string(),
        },
    )
}

fn push(coo: &mut CooMatrix, row: usize, col: usize, value: f64) {
    coo.push(row, col, value)
        .expect("indices were checked against the size line");
}
