//! Arithmetic on matrices and vectors.
//!
//! Every operation exists twice. The `try_*` methods return a
//! [`LinalgError::DimensionMismatch`] when the shapes do not fit. The operator traits
//! (`+`, `-`, `*`, unary `-`) are shorthand for code that knows its shapes: they call the
//! fallible method and panic with the same message when it fails.

use std::ops::{Add, Mul, Neg, Sub};

use crate::error::{LinalgError, Result};
use crate::matrix::Matrix;
use crate::sum;
use crate::vector::Vector;

impl Matrix {
    /// The element-wise sum of two matrices of the same shape.
    pub fn try_add(&self, other: &Matrix) -> Result<Matrix> {
        self.require_same_shape(other, "matrix addition")?;
        Ok(self.zip_with(other, |x, y| x + y))
    }

    /// The element-wise difference of two matrices of the same shape.
    pub fn try_sub(&self, other: &Matrix) -> Result<Matrix> {
        self.require_same_shape(other, "matrix subtraction")?;
        Ok(self.zip_with(other, |x, y| x - y))
    }

    /// The element-wise (Hadamard) product of two matrices of the same shape.
    pub fn hadamard(&self, other: &Matrix) -> Result<Matrix> {
        self.require_same_shape(other, "element-wise product")?;
        Ok(self.zip_with(other, |x, y| x * y))
    }

    /// The matrix product `self * other`; needs `self.cols() == other.rows()`.
    pub fn try_mul(&self, other: &Matrix) -> Result<Matrix> {
        if self.cols() != other.rows() {
            return Err(LinalgError::DimensionMismatch {
                op: "matrix product",
                left: self.shape(),
                right: other.shape(),
            });
        }
        let (n, m, p) = (self.rows(), self.cols(), other.cols());
        let mut out = vec![0.0; n * p];
        // The i-k-j order walks both the right operand and the output along rows.
        for i in 0..n {
            let out_row = &mut out[i * p..(i + 1) * p];
            for k in 0..m {
                let a = self.as_slice()[i * m + k];
                if a == 0.0 {
                    continue;
                }
                for (o, &b) in out_row.iter_mut().zip(other.row(k)) {
                    *o += a * b;
                }
            }
        }
        Ok(Matrix::from_parts(n, p, out))
    }

    /// The product of the matrix with a column vector; needs `self.cols() == v.len()`.
    pub fn try_mul_vec(&self, v: &Vector) -> Result<Vector> {
        if self.cols() != v.len() {
            return Err(LinalgError::DimensionMismatch {
                op: "matrix-vector product",
                left: self.shape(),
                right: (v.len(), 1),
            });
        }
        Ok(self
            .iter_rows()
            .map(|row| sum::dot(row, v.as_slice()))
            .collect())
    }

    /// A new matrix with every element multiplied by `factor`.
    pub fn scale(&self, factor: f64) -> Matrix {
        self.map(|x| x * factor)
    }

    fn zip_with<F: Fn(f64, f64) -> f64>(&self, other: &Matrix, f: F) -> Matrix {
        let data = self
            .as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(&x, &y)| f(x, y))
            .collect();
        Matrix::from_parts(self.rows(), self.cols(), data)
    }

    fn require_same_shape(&self, other: &Matrix, op: &'static str) -> Result<()> {
        if self.shape() == other.shape() {
            Ok(())
        } else {
            Err(LinalgError::DimensionMismatch {
                op,
                left: self.shape(),
                right: other.shape(),
            })
        }
    }
}

/// Unwraps the result of a fallible operation inside an operator implementation.
fn expect_shapes<T>(result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(err) => panic!("{}", err),
    }
}

/// Implements a binary operator for all four combinations of owned and borrowed operands
/// by forwarding to a fallible method on references.
macro_rules! forward_binary_operator {
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty, $out:ty, $try_method:ident) => {
        impl $trait<&$rhs> for &$lhs {
            type Output = $out;

            fn $method(self, rhs: &$rhs) -> $out {
                expect_shapes(self.$try_method(rhs))
            }
        }

        impl $trait<$rhs> for &$lhs {
            type Output = $out;

            fn $method(self, rhs: $rhs) -> $out {
                expect_shapes(self.$try_method(&rhs))
            }
        }

        impl $trait<&$rhs> for $lhs {
            type Output = $out;

            fn $method(self, rhs: &$rhs) -> $out {
                expect_shapes(self.$try_method(rhs))
            }
        }

        impl $trait<$rhs> for $lhs {
            type Output = $out;

            fn $method(self, rhs: $rhs) -> $out {
                expect_shapes(self.$try_method(&rhs))
            }
        }
    };
}

forward_binary_operator!(Add, add, Matrix, Matrix, Matrix, try_add);
forward_binary_operator!(Sub, sub, Matrix, Matrix, Matrix, try_sub);
forward_binary_operator!(Mul, mul, Matrix, Matrix, Matrix, try_mul);
forward_binary_operator!(Mul, mul, Matrix, Vector, Vector, try_mul_vec);
forward_binary_operator!(Add, add, Vector, Vector, Vector, try_add);
forward_binary_operator!(Sub, sub, Vector, Vector, Vector, try_sub);

/// Implements scaling by an `f64` on either side and negation, for owned and borrowed values.
macro_rules! scalar_operators {
    ($ty:ty) => {
        impl Mul<f64> for &$ty {
            type Output = $ty;

            fn mul(self, factor: f64) -> $ty {
                self.scale(factor)
            }
        }

        impl Mul<f64> for $ty {
            type Output = $ty;

            fn mul(self, factor: f64) -> $ty {
                self.scale(factor)
            }
        }

        impl Mul<&$ty> for f64 {
            type Output = $ty;

            fn mul(self, value: &$ty) -> $ty {
                value.scale(self)
            }
        }

        impl Mul<$ty> for f64 {
            type Output = $ty;

            fn mul(self, value: $ty) -> $ty {
                value.scale(self)
            }
        }

        impl Neg for &$ty {
            type Output = $ty;

            fn neg(self) -> $ty {
                self.scale(-1.0)
            }
        }

        impl Neg for $ty {
            type Output = $ty;

            fn neg(self) -> $ty {
                self.scale(-1.0)
            }
        }
    };
}

scalar_operators!(Matrix);
scalar_operators!(Vector);
