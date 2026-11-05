//! Sparse matrices.
//!
//! [`CooMatrix`] (coordinate form) is a list of `(row, column, value)` triplets: easy to
//! build, in any order, with repeated positions allowed.

mod coo;

pub use coo::CooMatrix;
