//! Reading matrices and vectors from files, with the format detected from the content.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::matrix::Matrix;
use crate::sparse::{CooMatrix, CsrMatrix};
use crate::vector::Vector;

use super::error::ParseError;
use super::{dense, mtx};

/// A matrix as it was stored in a file.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixData {
    /// The plain-text dense format.
    Dense(Matrix),
    /// The Matrix Market coordinate format.
    Sparse(CooMatrix),
}

impl MatrixData {
    /// The shape as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            MatrixData::Dense(matrix) => matrix.shape(),
            MatrixData::Sparse(coo) => coo.shape(),
        }
    }

    /// True for a matrix read from a sparse format.
    pub fn is_sparse(&self) -> bool {
        matches!(self, MatrixData::Sparse(_))
    }

    /// The matrix as a dense one.
    pub fn to_dense(&self) -> Matrix {
        match self {
            MatrixData::Dense(matrix) => matrix.clone(),
            MatrixData::Sparse(coo) => coo.to_dense(),
        }
    }

    /// The matrix in compressed sparse row form.
    pub fn to_csr(&self) -> CsrMatrix {
        match self {
            MatrixData::Dense(matrix) => CsrMatrix::from_dense(matrix),
            MatrixData::Sparse(coo) => coo.to_csr(),
        }
    }
}

/// Why a file could not be turned into a matrix or a vector.
#[derive(Debug, Clone, PartialEq)]
pub enum ReadError {
    /// The file could not be read at all.
    Io {
        path: PathBuf,
        kind: io::ErrorKind,
        message: String,
    },
    /// The file was read, but its content is not valid.
    Parse { path: PathBuf, error: ParseError },
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Io { path, message, .. } => {
                write!(f, "cannot read {}: {}", path.display(), message)
            }
            ReadError::Parse { path, error } => write!(f, "{}: {}", path.display(), error),
        }
    }
}

impl std::error::Error for ReadError {}

/// Parses a matrix in either format: Matrix Market when the text starts with the
/// `%%MatrixMarket` banner, the plain dense format otherwise.
pub fn parse_matrix_auto(text: &str) -> Result<MatrixData, ParseError> {
    if mtx::is_matrix_market(text) {
        mtx::parse_coordinate(text).map(MatrixData::Sparse)
    } else {
        dense::parse_matrix(text).map(MatrixData::Dense)
    }
}

/// Reads a matrix file in either format.
pub fn read_matrix<P: AsRef<Path>>(path: P) -> Result<MatrixData, ReadError> {
    let path = path.as_ref();
    let text = read_text(path)?;
    parse_matrix_auto(&text).map_err(|error| parse_error(path, error))
}

/// Reads a vector file in the plain dense format.
pub fn read_vector<P: AsRef<Path>>(path: P) -> Result<Vector, ReadError> {
    let path = path.as_ref();
    let text = read_text(path)?;
    dense::parse_vector(&text).map_err(|error| parse_error(path, error))
}

fn read_text(path: &Path) -> Result<String, ReadError> {
    fs::read_to_string(path).map_err(|error| ReadError::Io {
        path: path.to_path_buf(),
        kind: error.kind(),
        message: error.to_string(),
    })
}

fn parse_error(path: &Path, error: ParseError) -> ReadError {
    ReadError::Parse {
        path: path.to_path_buf(),
        error,
    }
}
