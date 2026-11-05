//! Dense and sparse linear algebra in plain Rust, with no dependencies beyond the standard
//! library.
//!
//! The dense side is built around [`Matrix`] (row-major `f64` storage) and [`Vector`].
//! Fallible operations come as `try_*` methods and functions that return
//! [`Result`]`<_, `[`LinalgError`]`>`; the arithmetic operators forward to them and panic
//! with the same message when the shapes do not fit.
//!
//! ```
//! use matrixlab::{Matrix, Vector};
//!
//! let a = Matrix::from_rows(&[[4.0, 1.0], [1.0, 3.0]]).unwrap();
//! let b = Vector::from_slice(&[1.0, 2.0]);
//!
//! let x = a.solve(&b).unwrap();
//! assert!((x[0] - 1.0 / 11.0).abs() < 1e-15);
//! assert!((x[1] - 7.0 / 11.0).abs() < 1e-15);
//! assert!((a.determinant().unwrap() - 11.0).abs() < 1e-14);
//!
//! // A singular matrix is an error value, not a panic.
//! let singular = Matrix::from_rows(&[[1.0, 2.0], [2.0, 4.0]]).unwrap();
//! assert!(singular.solve(&b).is_err());
//! ```
//!
//! # Modules
//!
//! | Module | Contents |
//! |---|---|
//! | [`matrix`], [`vector`], [`view`] | the dense types and borrowed block views |
//! | [`ops`], [`norms`], [`sum`] | arithmetic, norms and compensated summation |
//! | [`lu`], [`cholesky`], [`qr`] | decompositions, linear solves, least squares |
//! | [`eigen`] | Jacobi eigenvalue solver for symmetric matrices, power iteration |
//! | [`analysis`] | numerical rank and condition numbers |
//! | [`sparse`], [`iterative`] | COO and CSR matrices; Jacobi, Gauss-Seidel, conjugate gradient |
//! | [`io`], [`display`] | text formats with located parse errors; pretty printing |
//! | [`cli`] | the `matrixlab` command-line tool as a function |

pub mod analysis;
pub mod cholesky;
pub mod cli;
pub mod display;
pub mod eigen;
pub mod error;
pub mod io;
pub mod iterative;
pub mod lu;
pub mod matrix;
pub mod norms;
pub mod ops;
pub mod qr;
pub mod sparse;
pub mod sum;
pub mod vector;
pub mod view;

pub use error::{LinalgError, Result};
pub use matrix::Matrix;
pub use vector::Vector;
pub use view::MatrixView;

/// The version of the crate, as written in `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Compiles and runs the Rust examples of the README as documentation tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeExamples;
