//! Dense and sparse linear algebra in plain Rust.
//!
//! The crate uses the standard library only. Modules are added as the library grows.

pub mod error;
pub mod matrix;
pub mod sum;

pub use error::{LinalgError, Result};
pub use matrix::Matrix;

/// The version of the crate, as written in `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
