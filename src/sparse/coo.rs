//! The coordinate (triplet) format.

use crate::error::{LinalgError, Result};
use crate::matrix::Matrix;
use crate::vector::Vector;

/// A sparse matrix stored as a list of `(row, column, value)` triplets.
///
/// Triplets may come in any order and a position may appear more than once; repeated
/// entries add up, which is what assembling a matrix from element contributions needs.
/// Explicit zeros are kept.
#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix {
    rows: usize,
    cols: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

impl CooMatrix {
    /// An empty `rows` x `cols` matrix.
    pub fn new(rows: usize, cols: usize) -> CooMatrix {
        CooMatrix {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Builds a matrix from a list of triplets, checking every index.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Result<CooMatrix> {
        let mut coo = CooMatrix::new(rows, cols);
        for &(row, col, value) in triplets {
            coo.push(row, col, value)?;
        }
        Ok(coo)
    }

    /// Collects the non-zero elements of a dense matrix.
    pub fn from_dense(dense: &Matrix) -> CooMatrix {
        let mut coo = CooMatrix::new(dense.rows(), dense.cols());
        for (i, row) in dense.iter_rows().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    coo.row_indices.push(i);
                    coo.col_indices.push(j);
                    coo.values.push(value);
                }
            }
        }
        coo
    }

    /// Appends one entry, or fails with [`LinalgError::IndexOutOfBounds`].
    pub fn push(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(LinalgError::IndexOutOfBounds {
                index: (row, col),
                shape: self.shape(),
            });
        }
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
        Ok(())
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The shape as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The number of stored entries; repeated positions count once per entry.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// An iterator over the stored `(row, column, value)` triplets in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.values)
            .map(|((&row, &col), &value)| (row, col, value))
    }

    /// The transpose: every triplet with its indices exchanged.
    pub fn transpose(&self) -> CooMatrix {
        CooMatrix {
            rows: self.cols,
            cols: self.rows,
            row_indices: self.col_indices.clone(),
            col_indices: self.row_indices.clone(),
            values: self.values.clone(),
        }
    }

    /// The dense matrix with the same elements; repeated entries are added.
    pub fn to_dense(&self) -> Matrix {
        let mut dense = Matrix::zeros(self.rows, self.cols);
        for (row, col, value) in self.iter() {
            dense[(row, col)] += value;
        }
        dense
    }

    /// The product of the matrix with a vector.
    pub fn mul_vec(&self, x: &Vector) -> Result<Vector> {
        if x.len() != self.cols {
            return Err(LinalgError::DimensionMismatch {
                op: "sparse matrix-vector product",
                left: self.shape(),
                right: (x.len(), 1),
            });
        }
        let mut y = Vector::zeros(self.rows);
        for (row, col, value) in self.iter() {
            y[row] += value * x[col];
        }
        Ok(y)
    }
}
