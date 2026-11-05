//! Reading and writing matrices as text.
//!
//! [`dense`] is a plain whitespace- or comma-separated format with one row per line.
//! Every parser reports problems as a [`ParseError`] that names the line.

pub mod dense;
mod error;

pub use error::{ParseError, ParseErrorKind};
