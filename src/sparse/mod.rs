//! Sparse matrices.
//!
//! [`CooMatrix`] (coordinate form) is a list of `(row, column, value)` triplets: easy to
//! build, in any order, with repeated positions allowed. [`CsrMatrix`] (compressed sparse
//! row) is the form to compute with: rows are contiguous, columns are sorted and unique.
//! Build in COO, convert once with [`CooMatrix::to_csr`], then multiply and solve in CSR.

mod coo;
mod csr;

pub use coo::CooMatrix;
pub use csr::CsrMatrix;
