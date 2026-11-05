//! The error type shared by the numerical parts of the crate.

use std::fmt;

/// A convenient alias for results whose error is [`LinalgError`].
pub type Result<T> = std::result::Result<T, LinalgError>;

/// Everything that can go wrong in a numerical routine.
///
/// The variants carry the numbers needed to write a useful message: shapes, the index of the
/// pivot that failed, the residual reached when an iteration gave up, and so on.
#[derive(Debug, Clone, PartialEq)]
pub enum LinalgError {
    /// The number of elements does not match `rows * cols`.
    InvalidShape {
        rows: usize,
        cols: usize,
        len: usize,
    },
    /// A row has a different length from the first row.
    RaggedRows {
        row: usize,
        expected: usize,
        found: usize,
    },
    /// Two operands have shapes that do not fit the operation.
    DimensionMismatch {
        op: &'static str,
        left: (usize, usize),
        right: (usize, usize),
    },
    /// The operation needs a square matrix.
    NotSquare { rows: usize, cols: usize },
    /// An element or a range lies outside the matrix.
    IndexOutOfBounds {
        index: (usize, usize),
        shape: (usize, usize),
    },
    /// The operation is not defined for a matrix or vector without elements.
    Empty { op: &'static str },
    /// A pivot is zero or below the singularity tolerance.
    Singular { pivot: usize },
    /// The matrix is not symmetric; `row` and `col` name the first offending element.
    NotSymmetric { row: usize, col: usize },
    /// A Cholesky pivot or a conjugate-gradient curvature is not positive.
    NotPositiveDefinite { pivot: usize },
    /// A least-squares problem has linearly dependent columns.
    RankDeficient { column: usize },
    /// A least-squares problem has fewer rows than columns.
    Underdetermined { rows: usize, cols: usize },
    /// A stationary iteration needs a non-zero diagonal element in every row.
    ZeroDiagonal { row: usize },
    /// An iteration reached its limit, or diverged, before meeting its tolerance.
    NoConvergence {
        method: &'static str,
        iterations: usize,
        residual: f64,
    },
    /// The arrays describing a sparse matrix are inconsistent.
    InvalidStructure { reason: &'static str },
    /// An option has a value that makes no sense, for example a negative tolerance.
    InvalidArgument { what: &'static str },
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgError::InvalidShape { rows, cols, len } => match rows.checked_mul(*cols) {
                Some(needed) => write!(
                    f,
                    "a {}x{} matrix needs {} elements, but {} were given",
                    rows, cols, needed, len
                ),
                None => write!(
                    f,
                    "a {}x{} matrix has more elements than can be addressed",
                    rows, cols
                ),
            },
            LinalgError::RaggedRows {
                row,
                expected,
                found,
            } => write!(
                f,
                "row {} has {} elements, expected {}",
                row, found, expected
            ),
            LinalgError::DimensionMismatch { op, left, right } => write!(
                f,
                "dimension mismatch in {}: left is {}x{}, right is {}x{}",
                op, left.0, left.1, right.0, right.1
            ),
            LinalgError::NotSquare { rows, cols } => {
                write!(f, "matrix is {}x{}, but a square matrix is required", rows, cols)
            }
            LinalgError::IndexOutOfBounds { index, shape } => write!(
                f,
                "index ({}, {}) is out of bounds for a {}x{} matrix",
                index.0, index.1, shape.0, shape.1
            ),
            LinalgError::Empty { op } => write!(f, "{} is not defined for an empty input", op),
            LinalgError::Singular { pivot } => {
                write!(f, "matrix is singular to working precision (pivot {})", pivot)
            }
            LinalgError::NotSymmetric { row, col } => write!(
                f,
                "matrix is not symmetric: elements ({}, {}) and ({}, {}) differ",
                row, col, col, row
            ),
            LinalgError::NotPositiveDefinite { pivot } => {
                write!(f, "matrix is not positive definite (pivot {})", pivot)
            }
            LinalgError::RankDeficient { column } => write!(
                f,
                "matrix is rank deficient: column {} depends on the columns before it",
                column
            ),
            LinalgError::Underdetermined { rows, cols } => write!(
                f,
                "least squares needs at least as many rows as columns, got {}x{}",
                rows, cols
            ),
            LinalgError::ZeroDiagonal { row } => {
                write!(f, "diagonal element of row {} is zero", row)
            }
            LinalgError::NoConvergence {
                method,
                iterations,
                residual,
            } => write!(
                f,
                "{} did not converge after {} iterations (residual {:e})",
                method, iterations, residual
            ),
            LinalgError::InvalidStructure { reason } => {
                write!(f, "invalid sparse structure: {}", reason)
            }
            LinalgError::InvalidArgument { what } => write!(f, "invalid argument: {}", what),
        }
    }
}

impl std::error::Error for LinalgError {}
