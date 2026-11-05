//! Reading and writing matrices as text.
//!
//! [`dense`] is a plain whitespace- or comma-separated format with one row per line and
//! [`mtx`] is the Matrix Market coordinate format for sparse matrices. Every parser reports
//! problems as a [`ParseError`] that names the line. [`read_matrix`] and [`read_vector`]
//! read files and pick the matrix format from the content.

pub mod dense;
mod error;
mod file;
pub mod mtx;

pub use error::{ParseError, ParseErrorKind};
pub use file::{parse_matrix_auto, read_matrix, read_vector, MatrixData, ReadError};
