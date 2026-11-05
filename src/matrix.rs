//! The dense matrix type.

use std::ops::{Index, IndexMut};

use crate::error::{LinalgError, Result};

/// A dense matrix of `f64` values stored row by row in one contiguous buffer.
///
/// Element `(i, j)` lives at offset `i * cols + j`. A matrix may have zero rows or zero
/// columns; routines that cannot work on such a matrix say so with
/// [`LinalgError::Empty`].
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Creates a `rows` x `cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix::filled(rows, cols, 0.0)
    }

    /// Creates a `rows` x `cols` matrix with every element equal to `value`.
    pub fn filled(rows: usize, cols: usize, value: f64) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    /// Creates the `n` x `n` identity matrix.
    pub fn identity(n: usize) -> Matrix {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Creates a square matrix with `diagonal` on its main diagonal and zeros elsewhere.
    pub fn from_diagonal(diagonal: &[f64]) -> Matrix {
        let n = diagonal.len();
        let mut m = Matrix::zeros(n, n);
        for (i, &d) in diagonal.iter().enumerate() {
            m.data[i * n + i] = d;
        }
        m
    }

    /// Builds a matrix from row-major data.
    ///
    /// Fails with [`LinalgError::InvalidShape`] when `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Matrix> {
        if data.len() != rows * cols {
            return Err(LinalgError::InvalidShape {
                rows,
                cols,
                len: data.len(),
            });
        }
        Ok(Matrix { rows, cols, data })
    }

    /// Builds a matrix from a list of rows.
    ///
    /// Fails with [`LinalgError::RaggedRows`] when a row is shorter or longer than the
    /// first one. An empty list gives a 0 x 0 matrix.
    pub fn from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Result<Matrix> {
        let cols = rows.first().map_or(0, |r| r.as_ref().len());
        let mut data = Vec::with_capacity(rows.len() * cols);
        for (i, row) in rows.iter().enumerate() {
            let row = row.as_ref();
            if row.len() != cols {
                return Err(LinalgError::RaggedRows {
                    row: i,
                    expected: cols,
                    found: row.len(),
                });
            }
            data.extend_from_slice(row);
        }
        Ok(Matrix {
            rows: rows.len(),
            cols,
            data,
        })
    }

    /// Builds a matrix by calling `f(i, j)` for every element.
    pub fn from_fn<F: FnMut(usize, usize) -> f64>(rows: usize, cols: usize, mut f: F) -> Matrix {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Matrix { rows, cols, data }
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

    /// True when the matrix has as many rows as columns.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// True when the matrix has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The elements in row-major order.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The elements in row-major order, mutable.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Consumes the matrix and returns its row-major buffer.
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    /// The element at `(row, col)`, or `None` when it is out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// The element at `(row, col)`, or [`LinalgError::IndexOutOfBounds`].
    pub fn try_get(&self, row: usize, col: usize) -> Result<f64> {
        self.get(row, col).ok_or(LinalgError::IndexOutOfBounds {
            index: (row, col),
            shape: self.shape(),
        })
    }

    /// Stores `value` at `(row, col)`, or fails with [`LinalgError::IndexOutOfBounds`].
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = value;
            Ok(())
        } else {
            Err(LinalgError::IndexOutOfBounds {
                index: (row, col),
                shape: self.shape(),
            })
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    /// Panics when the index is out of bounds; use [`Matrix::get`] to avoid the panic.
    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        assert!(
            row < self.rows && col < self.cols,
            "index ({}, {}) is out of bounds for a {}x{} matrix",
            row,
            col,
            self.rows,
            self.cols
        );
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        assert!(
            row < self.rows && col < self.cols,
            "index ({}, {}) is out of bounds for a {}x{} matrix",
            row,
            col,
            self.rows,
            self.cols
        );
        &mut self.data[row * self.cols + col]
    }
}
