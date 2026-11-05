//! Dense and sparse linear algebra in plain Rust.
//!
//! The crate uses the standard library only. Modules are added as the library grows.

pub mod cholesky;
pub mod error;
pub mod lu;
pub mod matrix;
pub mod norms;
pub mod ops;
pub mod qr;
pub mod sum;
pub mod vector;
pub mod view;

pub use error::{LinalgError, Result};
pub use matrix::Matrix;
pub use vector::Vector;
pub use view::MatrixView;

/// The version of the crate, as written in `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
