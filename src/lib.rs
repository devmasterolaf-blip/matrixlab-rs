//! Dense and sparse linear algebra in plain Rust.
//!
//! The crate uses the standard library only. Modules are added as the library grows.

pub mod error;
pub mod matrix;
pub mod norms;
pub mod ops;
pub mod sum;
pub mod vector;

pub use error::{LinalgError, Result};
pub use matrix::Matrix;
pub use vector::Vector;

/// The version of the crate, as written in `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
