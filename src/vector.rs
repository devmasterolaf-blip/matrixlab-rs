//! The dense vector type.

use std::ops::{Index, IndexMut};

use crate::error::{LinalgError, Result};
use crate::sum;

/// A dense vector of `f64` values.
///
/// It has no orientation: the matrix routines treat it as a column on the right of a
/// product and as the right-hand side of a linear system.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    /// Wraps a buffer.
    pub fn new(data: Vec<f64>) -> Vector {
        Vector { data }
    }

    /// A vector of `n` zeros.
    pub fn zeros(n: usize) -> Vector {
        Vector { data: vec![0.0; n] }
    }

    /// A vector with every element equal to `value`.
    pub fn filled(n: usize, value: f64) -> Vector {
        Vector {
            data: vec![value; n],
        }
    }

    /// The unit vector of length `n` with a one at position `i`. Panics when `i >= n`.
    pub fn unit(n: usize, i: usize) -> Vector {
        assert!(i < n, "unit vector index {} is out of bounds for length {}", i, n);
        let mut v = Vector::zeros(n);
        v.data[i] = 1.0;
        v
    }

    /// Copies a slice.
    pub fn from_slice(data: &[f64]) -> Vector {
        Vector {
            data: data.to_vec(),
        }
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the vector has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The elements as a slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The elements as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Consumes the vector and returns its buffer.
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    /// An iterator over the elements.
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.data.iter()
    }

    /// The element at `i`, or `None` when it is out of bounds.
    pub fn get(&self, i: usize) -> Option<f64> {
        self.data.get(i).copied()
    }

    /// The dot product with `other`, summed with compensation.
    pub fn dot(&self, other: &Vector) -> Result<f64> {
        self.require_same_len(other, "dot product")?;
        Ok(sum::dot(&self.data, &other.data))
    }

    /// The sum of the absolute values.
    pub fn norm1(&self) -> f64 {
        sum::sum(self.data.iter().map(|x| x.abs()))
    }

    /// The Euclidean length, safe against overflow of the squares.
    pub fn norm2(&self) -> f64 {
        sum::norm2(&self.data)
    }

    /// The largest absolute value; zero for an empty vector, NaN when an element is NaN.
    pub fn norm_inf(&self) -> f64 {
        let mut max = 0.0_f64;
        for &x in &self.data {
            if x.is_nan() {
                return f64::NAN;
            }
            max = max.max(x.abs());
        }
        max
    }

    /// A new vector with every element multiplied by `factor`.
    pub fn scale(&self, factor: f64) -> Vector {
        Vector {
            data: self.data.iter().map(|&x| x * factor).collect(),
        }
    }

    /// The element-wise sum, or [`LinalgError::DimensionMismatch`].
    pub fn try_add(&self, other: &Vector) -> Result<Vector> {
        self.require_same_len(other, "vector addition")?;
        Ok(self.zip_with(other, |x, y| x + y))
    }

    /// The element-wise difference, or [`LinalgError::DimensionMismatch`].
    pub fn try_sub(&self, other: &Vector) -> Result<Vector> {
        self.require_same_len(other, "vector subtraction")?;
        Ok(self.zip_with(other, |x, y| x - y))
    }

    /// Adds `alpha * x` to this vector in place.
    pub fn axpy(&mut self, alpha: f64, x: &Vector) -> Result<()> {
        self.require_same_len(x, "axpy")?;
        for (y, &xi) in self.data.iter_mut().zip(&x.data) {
            *y += alpha * xi;
        }
        Ok(())
    }

    /// The vector divided by its Euclidean length.
    ///
    /// Fails with [`LinalgError::Empty`] for a vector without elements and with
    /// [`LinalgError::InvalidArgument`] for a zero vector, which has no direction.
    pub fn normalized(&self) -> Result<Vector> {
        if self.data.is_empty() {
            return Err(LinalgError::Empty { op: "normalization" });
        }
        let norm = self.norm2();
        if norm == 0.0 {
            return Err(LinalgError::InvalidArgument {
                what: "the zero vector cannot be normalized",
            });
        }
        Ok(self.scale(1.0 / norm))
    }

    fn zip_with<F: Fn(f64, f64) -> f64>(&self, other: &Vector, f: F) -> Vector {
        Vector {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&x, &y)| f(x, y))
                .collect(),
        }
    }

    fn require_same_len(&self, other: &Vector, op: &'static str) -> Result<()> {
        if self.len() == other.len() {
            Ok(())
        } else {
            Err(LinalgError::DimensionMismatch {
                op,
                left: (self.len(), 1),
                right: (other.len(), 1),
            })
        }
    }
}

impl From<Vec<f64>> for Vector {
    fn from(data: Vec<f64>) -> Vector {
        Vector { data }
    }
}

impl From<&[f64]> for Vector {
    fn from(data: &[f64]) -> Vector {
        Vector::from_slice(data)
    }
}

impl FromIterator<f64> for Vector {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Vector {
        Vector {
            data: iter.into_iter().collect(),
        }
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}
