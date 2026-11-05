//! Human-readable formatting of matrices and vectors.

use std::fmt;

use crate::matrix::Matrix;
use crate::vector::Vector;

/// The number of digits after the decimal point used by the `Display` implementations when
/// the format string does not ask for a precision.
pub const DEFAULT_PRECISION: usize = 4;

/// How numbers are written by [`format_matrix`] and [`format_vector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatOptions {
    /// The number of digits after the decimal point.
    pub precision: usize,
    /// Write every number in scientific notation (`1.2500e2`) instead of fixed notation.
    pub scientific: bool,
}

impl Default for FormatOptions {
    fn default() -> FormatOptions {
        FormatOptions {
            precision: DEFAULT_PRECISION,
            scientific: false,
        }
    }
}

/// Formats one number.
///
/// A value that rounds to zero is written without a minus sign, so rounding noise such as
/// `-1e-17` does not show up as `-0.0000`.
pub fn format_number(value: f64, options: &FormatOptions) -> String {
    let text = if options.scientific {
        format!("{:.*e}", options.precision, value)
    } else {
        format!("{:.*}", options.precision, value)
    };
    match text.strip_prefix('-') {
        Some(rest) if rest.chars().all(|c| matches!(c, '0' | '.' | 'e')) => rest.to_string(),
        _ => text,
    }
}

/// Formats a matrix as one bracketed line per row, with every column right-aligned.
///
/// ```
/// use matrixlab::display::{format_matrix, FormatOptions};
/// use matrixlab::Matrix;
///
/// let m = Matrix::from_rows(&[[1.0, -2.5], [10.0, 3.0]]).unwrap();
/// let options = FormatOptions { precision: 2, scientific: false };
/// assert_eq!(format_matrix(&m, &options), "[  1.00  -2.50 ]\n[ 10.00   3.00 ]");
/// ```
///
/// A matrix without rows is written as `[]`.
pub fn format_matrix(matrix: &Matrix, options: &FormatOptions) -> String {
    if matrix.rows() == 0 {
        return "[]".to_string();
    }
    let cells: Vec<Vec<String>> = matrix
        .iter_rows()
        .map(|row| row.iter().map(|&x| format_number(x, options)).collect())
        .collect();
    let mut widths = vec![0; matrix.cols()];
    for row in &cells {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.len());
        }
    }
    let lines: Vec<String> = cells
        .iter()
        .map(|row| {
            let padded: Vec<String> = row
                .iter()
                .zip(&widths)
                .map(|(cell, &width)| format!("{:>width$}", cell, width = width))
                .collect();
            bracket(&padded)
        })
        .collect();
    lines.join("\n")
}

/// Formats a vector on one line, for example `[ 1.0000  2.0000 ]`.
pub fn format_vector(vector: &Vector, options: &FormatOptions) -> String {
    let cells: Vec<String> = vector
        .iter()
        .map(|&x| format_number(x, options))
        .collect();
    bracket(&cells)
}

fn bracket(cells: &[String]) -> String {
    if cells.is_empty() {
        "[ ]".to_string()
    } else {
        format!("[ {} ]", cells.join("  "))
    }
}

fn options_from(f: &fmt::Formatter<'_>) -> FormatOptions {
    FormatOptions {
        precision: f.precision().unwrap_or(DEFAULT_PRECISION),
        scientific: false,
    }
}

impl fmt::Display for Matrix {
    /// Writes the matrix with [`format_matrix`]; `{:.2}` selects two decimals.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_matrix(self, &options_from(f)))
    }
}

impl fmt::Display for Vector {
    /// Writes the vector with [`format_vector`]; `{:.2}` selects two decimals.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_vector(self, &options_from(f)))
    }
}
