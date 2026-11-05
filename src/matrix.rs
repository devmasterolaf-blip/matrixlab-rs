//! The dense matrix type.

use std::ops::{Index, IndexMut};

use crate::error::{LinalgError, Result};
use crate::sum;

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
    /// Fails with [`LinalgError::InvalidShape`] when `data.len() != rows * cols`, which
    /// includes shapes whose element count does not fit in a `usize`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Matrix> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(LinalgError::InvalidShape {
                rows,
                cols,
                len: data.len(),
            });
        }
        Ok(Matrix { rows, cols, data })
    }

    /// Wraps a buffer whose length the caller has already made equal to `rows * cols`.
    pub(crate) fn from_parts(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        debug_assert_eq!(data.len(), rows * cols);
        Matrix { rows, cols, data }
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

    /// Row `i` as a slice. Panics when `i` is out of bounds.
    pub fn row(&self, i: usize) -> &[f64] {
        assert!(i < self.rows, "row {} is out of bounds for {} rows", i, self.rows);
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Row `i` as a mutable slice. Panics when `i` is out of bounds.
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        assert!(i < self.rows, "row {} is out of bounds for {} rows", i, self.rows);
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// An iterator over the rows, each one a slice.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[f64]> + '_ {
        (0..self.rows).map(move |i| &self.data[i * self.cols..(i + 1) * self.cols])
    }

    /// A copy of column `j`. Panics when `j` is out of bounds.
    pub fn column(&self, j: usize) -> Vec<f64> {
        assert!(j < self.cols, "column {} is out of bounds for {} columns", j, self.cols);
        (0..self.rows).map(|i| self.data[i * self.cols + j]).collect()
    }

    /// A copy of the main diagonal; for a rectangular matrix it has `min(rows, cols)` elements.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.rows.min(self.cols))
            .map(|i| self.data[i * self.cols + i])
            .collect()
    }

    /// The sum of the diagonal elements of a square matrix, accumulated with compensated
    /// summation. The trace of a 0 x 0 matrix is zero.
    pub fn trace(&self) -> Result<f64> {
        self.require_square()?;
        Ok(sum::sum(self.diagonal()))
    }

    /// The transpose, as a new matrix.
    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        t
    }

    /// A new matrix with `f` applied to every element.
    pub fn map<F: FnMut(f64) -> f64>(&self, mut f: F) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Exchanges rows `a` and `b` in place. Panics when either is out of bounds.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        assert!(
            a < self.rows && b < self.rows,
            "cannot swap rows {} and {} of a matrix with {} rows",
            a,
            b,
            self.rows
        );
        if a == b {
            return;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let (head, tail) = self.data.split_at_mut(hi * self.cols);
        head[lo * self.cols..(lo + 1) * self.cols].swap_with_slice(&mut tail[..self.cols]);
    }

    /// True when the matrix is square and `|a[i][j] - a[j][i]| <= tol` for every pair.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        self.first_asymmetry(tol).is_none() && self.is_square()
    }

    /// The first `(row, col)` above the diagonal whose mirror element differs by more than
    /// `tol`. Only the leading square part of a rectangular matrix is inspected.
    pub(crate) fn first_asymmetry(&self, tol: f64) -> Option<(usize, usize)> {
        let n = self.rows.min(self.cols);
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = (self.data[i * self.cols + j] - self.data[j * self.cols + i]).abs();
                // A NaN difference is an asymmetry as well, hence the negated comparison.
                if !(diff <= tol) {
                    return Some((i, j));
                }
            }
        }
        None
    }

    /// Returns [`LinalgError::NotSquare`] unless the matrix is square.
    pub(crate) fn require_square(&self) -> Result<()> {
        if self.is_square() {
            Ok(())
        } else {
            Err(LinalgError::NotSquare {
                rows: self.rows,
                cols: self.cols,
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
