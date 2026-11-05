//! The compressed sparse row format.

use crate::error::{LinalgError, Result};
use crate::matrix::Matrix;
use crate::sum;
use crate::vector::Vector;

use super::coo::CooMatrix;

/// A sparse matrix in compressed sparse row form.
///
/// The entries of row `i` are `values[row_offsets[i]..row_offsets[i + 1]]`, with their
/// column numbers at the same positions of `col_indices`. Within a row the columns are
/// strictly increasing, so a position is stored at most once and can be found by binary
/// search.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    rows: usize,
    cols: usize,
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// Builds a matrix from the three CSR arrays after checking that they are consistent.
    ///
    /// Fails with [`LinalgError::InvalidStructure`] when `row_offsets` does not have
    /// `rows + 1` elements, does not start at zero, decreases or does not end at the number
    /// of entries; when the two entry arrays differ in length; or when the columns of a row
    /// are out of range or not strictly increasing.
    pub fn from_raw_parts(
        rows: usize,
        cols: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<CsrMatrix> {
        let invalid = |reason| Err(LinalgError::InvalidStructure { reason });
        if row_offsets.len() != rows + 1 {
            return invalid("row_offsets must have rows + 1 elements");
        }
        if col_indices.len() != values.len() {
            return invalid("col_indices and values must have the same length");
        }
        if row_offsets[0] != 0 {
            return invalid("row_offsets must start at zero");
        }
        if row_offsets[rows] != values.len() {
            return invalid("row_offsets must end at the number of entries");
        }
        for i in 0..rows {
            let (start, end) = (row_offsets[i], row_offsets[i + 1]);
            if start > end || end > values.len() {
                return invalid("row_offsets must not decrease");
            }
            let columns = &col_indices[start..end];
            if columns.iter().any(|&c| c >= cols) {
                return invalid("a column index is out of range");
            }
            if columns.windows(2).any(|pair| pair[0] >= pair[1]) {
                return invalid("column indices must be strictly increasing within a row");
            }
        }
        Ok(CsrMatrix {
            rows,
            cols,
            row_offsets,
            col_indices,
            values,
        })
    }

    /// Compresses a coordinate matrix: sorts each row by column and adds repeated entries.
    pub fn from_coo(coo: &CooMatrix) -> CsrMatrix {
        let (rows, cols) = coo.shape();
        // Counting sort by row, which keeps the insertion order inside each row.
        let mut row_offsets = vec![0; rows + 1];
        for (row, _, _) in coo.iter() {
            row_offsets[row + 1] += 1;
        }
        for i in 0..rows {
            row_offsets[i + 1] += row_offsets[i];
        }
        let mut next = row_offsets.clone();
        let mut entries = vec![(0, 0.0); coo.nnz()];
        for (row, col, value) in coo.iter() {
            entries[next[row]] = (col, value);
            next[row] += 1;
        }

        let mut col_indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        let mut merged_offsets = vec![0; rows + 1];
        for i in 0..rows {
            let row_entries = &mut entries[row_offsets[i]..row_offsets[i + 1]];
            row_entries.sort_by_key(|&(col, _)| col);
            let row_start = col_indices.len();
            for &(col, value) in row_entries.iter() {
                if col_indices.len() > row_start && col_indices.last() == Some(&col) {
                    *values.last_mut().expect("values and col_indices grow together") += value;
                } else {
                    col_indices.push(col);
                    values.push(value);
                }
            }
            merged_offsets[i + 1] = col_indices.len();
        }
        CsrMatrix {
            rows,
            cols,
            row_offsets: merged_offsets,
            col_indices,
            values,
        }
    }

    /// Collects the non-zero elements of a dense matrix.
    pub fn from_dense(dense: &Matrix) -> CsrMatrix {
        CsrMatrix::from_coo(&CooMatrix::from_dense(dense))
    }

    /// The `n` x `n` identity matrix.
    pub fn identity(n: usize) -> CsrMatrix {
        CsrMatrix {
            rows: n,
            cols: n,
            row_offsets: (0..=n).collect(),
            col_indices: (0..n).collect(),
            values: vec![1.0; n],
        }
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

    /// The number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// The row offsets array, of length `rows + 1`.
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// The column index of every stored entry, row by row.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// The value of every stored entry, row by row.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The stored columns and values of row `i`. Panics when `i` is out of bounds.
    pub fn row(&self, i: usize) -> (&[usize], &[f64]) {
        assert!(i < self.rows, "row {} is out of bounds for {} rows", i, self.rows);
        let range = self.row_offsets[i]..self.row_offsets[i + 1];
        (&self.col_indices[range.clone()], &self.values[range])
    }

    /// The element at `(row, col)`: `None` outside the matrix, zero for a position that is
    /// not stored.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let (columns, values) = self.row(row);
        Some(match columns.binary_search(&col) {
            Ok(position) => values[position],
            Err(_) => 0.0,
        })
    }

    /// The main diagonal, with zeros for positions that are not stored.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.rows.min(self.cols))
            .map(|i| self.get(i, i).unwrap_or(0.0))
            .collect()
    }

    /// An iterator over the stored `(row, column, value)` triplets in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.rows).flat_map(move |i| {
            let (columns, values) = self.row(i);
            columns.iter().zip(values).map(move |(&j, &v)| (i, j, v))
        })
    }

    /// The product of the matrix with a vector, with compensated row sums.
    pub fn mul_vec(&self, x: &Vector) -> Result<Vector> {
        if x.len() != self.cols {
            return Err(LinalgError::DimensionMismatch {
                op: "sparse matrix-vector product",
                left: self.shape(),
                right: (x.len(), 1),
            });
        }
        Ok((0..self.rows)
            .map(|i| {
                let (columns, values) = self.row(i);
                sum::sum(columns.iter().zip(values).map(|(&j, &v)| v * x[j]))
            })
            .collect())
    }

    /// The transpose, also in CSR form.
    pub fn transpose(&self) -> CsrMatrix {
        CsrMatrix::from_coo(&self.to_coo().transpose())
    }

    /// The same matrix as a list of triplets in row-major order.
    pub fn to_coo(&self) -> CooMatrix {
        let mut coo = CooMatrix::new(self.rows, self.cols);
        for (row, col, value) in self.iter() {
            coo.push(row, col, value)
                .expect("a CSR matrix only stores in-range entries");
        }
        coo
    }

    /// The dense matrix with the same elements.
    pub fn to_dense(&self) -> Matrix {
        let mut dense = Matrix::zeros(self.rows, self.cols);
        for (row, col, value) in self.iter() {
            dense[(row, col)] = value;
        }
        dense
    }

    /// True when the matrix is square and every element differs from its mirror element by
    /// at most `tol`.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        self.iter().all(|(row, col, value)| {
            let mirror = self.get(col, row).unwrap_or(0.0);
            (value - mirror).abs() <= tol
        })
    }
}

impl CooMatrix {
    /// Compresses the matrix into CSR form; see [`CsrMatrix::from_coo`].
    pub fn to_csr(&self) -> CsrMatrix {
        CsrMatrix::from_coo(self)
    }
}
