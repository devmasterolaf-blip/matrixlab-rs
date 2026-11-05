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
        }
    }
}

impl std::error::Error for ParseError {}
