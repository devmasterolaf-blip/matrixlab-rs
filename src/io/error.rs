//! Errors of the text formats.

use std::fmt;

/// A problem found while parsing text, with the line it was found on.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// The 1-based number of the offending line. For problems that only show at the end of
    /// the input (nothing to read, too few entries) it is the number of the last line.
    pub line: usize,
    /// What is wrong.
    pub kind: ParseErrorKind,
}

/// The kinds of problems the parsers report.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    /// The input has no data: it is empty or made of comments and blank lines only.
    EmptyInput,
    /// A field is not a number. `column` is the 1-based position of the field in its line.
    InvalidNumber { column: usize, token: String },
    /// A row has a different number of fields from the first row.
    RaggedRow { expected: usize, found: usize },
    /// A vector was expected, but the data has several rows and several columns.
    NotAVector { rows: usize, cols: usize },
    /// A Matrix Market file does not start with a `%%MatrixMarket` banner.
    MissingBanner,
    /// The banner does not have the five words `%%MatrixMarket object format field symmetry`.
    MalformedBanner { found: usize },
    /// A banner word names something this crate does not read. `what` is the role of the
    /// word (`"object"`, `"format"`, `"field"` or `"symmetry"`).
    Unsupported { what: &'static str, token: String },
    /// A line has the wrong number of fields.
    FieldCount { expected: usize, found: usize },
    /// A size or an index is not a non-negative whole number.
    InvalidInteger { column: usize, token: String },
    /// A 1-based index is zero or larger than the dimension it addresses.
    IndexOutOfRange { index: usize, limit: usize },
    /// The number of entry lines differs from the count announced in the size line.
    EntryCount { expected: usize, found: usize },
    /// A symmetric or skew-symmetric file describes a matrix that is not square.
    SymmetryNeedsSquare { rows: usize, cols: usize },
    /// A skew-symmetric file has an entry on the diagonal.
    SkewDiagonalEntry { index: usize },
}

impl ParseError {
    pub(crate) fn new(line: usize, kind: ParseErrorKind) -> ParseError {
        ParseError { line, kind }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.kind)
    }
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::EmptyInput => write!(f, "the input contains no data"),
            ParseErrorKind::InvalidNumber { column, token } => {
                write!(f, "field {} is not a number: '{}'", column, token)
            }
            ParseErrorKind::RaggedRow { expected, found } => {
                write!(f, "expected {} fields, found {}", expected, found)
            }
            ParseErrorKind::NotAVector { rows, cols } => write!(
                f,
                "expected a single row or a single column, found a {}x{} matrix",
                rows, cols
            ),
            ParseErrorKind::MissingBanner => {
                write!(f, "the file does not start with a %%MatrixMarket banner")
            }
            ParseErrorKind::MalformedBanner { found } => {
                write!(f, "the banner needs 5 words, found {}", found)
            }
            ParseErrorKind::Unsupported { what, token } => {
                write!(f, "unsupported {} '{}'", what, token)
            }
            ParseErrorKind::FieldCount { expected, found } => {
                write!(f, "expected {} fields, found {}", expected, found)
            }
            ParseErrorKind::InvalidInteger { column, token } => write!(
                f,
                "field {} is not a non-negative whole number: '{}'",
                column, token
            ),
            ParseErrorKind::IndexOutOfRange { index, limit } => {
                write!(f, "index {} is outside 1..={}", index, limit)
            }
            ParseErrorKind::EntryCount { expected, found } => write!(
                f,
                "the size line announces {} entries, but {} were found",
                expected, found
            ),
            ParseErrorKind::SymmetryNeedsSquare { rows, cols } => write!(
                f,
                "a symmetric matrix must be square, but the size line says {}x{}",
                rows, cols
            ),
            ParseErrorKind::SkewDiagonalEntry { index } => write!(
                f,
                "a skew-symmetric matrix cannot have an entry on its diagonal ({}, {})",
                index, index
            ),
        }
    }
}

impl std::error::Error for ParseError {}
