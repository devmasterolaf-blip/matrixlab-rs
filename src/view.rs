//! Read-only views of a rectangular block of a matrix.

use std::ops::{Index, Range};

use crate::error::{LinalgError, Result};
use crate::matrix::Matrix;
use crate::sum;
use crate::vector::Vector;

/// A borrowed, read-only window onto a block of a [`Matrix`].
///
/// A view costs nothing to create: it stores the borrowed matrix and the position of the
/// block. A single row, a single column and a sub-block are all the same type.
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a> {
    matrix: &'a Matrix,
    row0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
}

impl<'a> MatrixView<'a> {
    /// The number of rows in the view.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns in the view.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The shape of the view as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The element at `(row, col)` of the view, or `None` when it is outside the view.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.rows && col < self.cols {
            Some(self.matrix[(self.row0 + row, self.col0 + col)])
        } else {
            None
        }
    }

    /// Row `i` of the view as a slice of the underlying matrix. Panics when out of bounds.
    pub fn row(&self, i: usize) -> &'a [f64] {
        assert!(i < self.rows, "row {} is out of bounds for {} rows", i, self.rows);
        &self.matrix.row(self.row0 + i)[self.col0..self.col0 + self.cols]
    }

    /// An iterator over the rows of the view.
    pub fn iter_rows(&self) -> impl Iterator<Item = &'a [f64]> + '_ {
        (0..self.rows).map(move |i| self.row(i))
    }

    /// A copy of column `j` of the view. Panics when out of bounds.
    pub fn column(&self, j: usize) -> Vec<f64> {
        assert!(j < self.cols, "column {} is out of bounds for {} columns", j, self.cols);
        (0..self.rows).map(|i| self.row(i)[j]).collect()
    }

    /// A narrower view, with ranges relative to this view.
    pub fn view(&self, rows: Range<usize>, cols: Range<usize>) -> Result<MatrixView<'a>> {
        check_ranges(&rows, &cols, self.shape())?;
        Ok(MatrixView {
            matrix: self.matrix,
            row0: self.row0 + rows.start,
            col0: self.col0 + cols.start,
            rows: rows.end - rows.start,
            cols: cols.end - cols.start,
        })
    }

    /// Copies the block into a matrix of its own.
    pub fn to_matrix(&self) -> Matrix {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for row in self.iter_rows() {
            data.extend_from_slice(row);
        }
        Matrix::from_parts(self.rows, self.cols, data)
    }

    /// Copies a view that is a single row or a single column into a vector.
    ///
    /// Fails with [`LinalgError::DimensionMismatch`] when the view has more than one row
    /// and more than one column.
    pub fn to_vector(&self) -> Result<Vector> {
        if self.rows != 1 && self.cols != 1 {
            return Err(LinalgError::DimensionMismatch {
                op: "conversion of a view to a vector",
                left: self.shape(),
                right: (self.rows.max(self.cols), 1),
            });
        }
        Ok(self.iter_rows().flatten().copied().collect())
    }

    /// The compensated sum of the elements.
    pub fn sum(&self) -> f64 {
        sum::sum(self.iter_rows().flatten().copied())
    }

    /// The Frobenius norm of the block.
    pub fn norm_frobenius(&self) -> f64 {
        self.to_matrix().norm_frobenius()
    }
}

impl Index<(usize, usize)> for MatrixView<'_> {
    type Output = f64;

    /// Panics when the index is outside the view.
    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        assert!(
            row < self.rows && col < self.cols,
            "index ({}, {}) is out of bounds for a {}x{} view",
            row,
            col,
            self.rows,
            self.cols
        );
        &self.matrix[(self.row0 + row, self.col0 + col)]
    }
}

impl Matrix {
    /// A view of the whole matrix.
    pub fn as_view(&self) -> MatrixView<'_> {
        MatrixView {
            matrix: self,
            row0: 0,
            col0: 0,
            rows: self.rows(),
            cols: self.cols(),
        }
    }

    /// A view of the block made of the given row and column ranges.
    ///
    /// Fails with [`LinalgError::IndexOutOfBounds`] when a range ends past the matrix and
    /// with [`LinalgError::InvalidArgument`] when a range runs backwards.
    pub fn view(&self, rows: Range<usize>, cols: Range<usize>) -> Result<MatrixView<'_>> {
        self.as_view().view(rows, cols)
    }

    /// A view of row `i`, of shape `1 x cols`.
    pub fn row_view(&self, i: usize) -> Result<MatrixView<'_>> {
        if i >= self.rows() {
            return Err(LinalgError::IndexOutOfBounds {
                index: (i, 0),
                shape: self.shape(),
            });
        }
        self.view(i..i + 1, 0..self.cols())
    }

    /// A view of column `j`, of shape `rows x 1`.
    pub fn column_view(&self, j: usize) -> Result<MatrixView<'_>> {
        if j >= self.cols() {
            return Err(LinalgError::IndexOutOfBounds {
                index: (0, j),
                shape: self.shape(),
            });
        }
        self.view(0..self.rows(), j..j + 1)
    }

    /// A copy of the block made of the given row and column ranges.
    pub fn submatrix(&self, rows: Range<usize>, cols: Range<usize>) -> Result<Matrix> {
        Ok(self.view(rows, cols)?.to_matrix())
    }

    /// Overwrites the block whose top-left corner is `(row0, col0)` with `block`.
    ///
    /// Fails with [`LinalgError::IndexOutOfBounds`] when the block does not fit.
    pub fn set_block(&mut self, row0: usize, col0: usize, block: &Matrix) -> Result<()> {
        let end = (row0 + block.rows(), col0 + block.cols());
        if end.0 > self.rows() || end.1 > self.cols() {
            return Err(LinalgError::IndexOutOfBounds {
                index: end,
                shape: self.shape(),
            });
        }
        for (i, source) in block.iter_rows().enumerate() {
            self.row_mut(row0 + i)[col0..end.1].copy_from_slice(source);
        }
        Ok(())
    }
}

fn check_ranges(rows: &Range<usize>, cols: &Range<usize>, shape: (usize, usize)) -> Result<()> {
    if rows.start > rows.end || cols.start > cols.end {
        return Err(LinalgError::InvalidArgument {
            what: "a range must not start after its end",
        });
    }
    if rows.end > shape.0 || cols.end > shape.1 {
        return Err(LinalgError::IndexOutOfBounds {
            index: (rows.end, cols.end),
            shape,
        });
    }
    Ok(())
}
