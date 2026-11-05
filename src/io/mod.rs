//! Reading and writing matrices as text.
//!
//! [`dense`] is a plain whitespace- or comma-separated format with one row per line and
//! [`mtx`] is the Matrix Market coordinate format for sparse matrices. Every parser reports
//! problems as a [`ParseError`] that names the line.

pub mod dense;
mod error;
pub mod mtx;

pub use error::{ParseError, ParseErrorKind};
