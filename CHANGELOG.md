# Changelog

All notable changes to this project are recorded in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2026-11-05

The first version.

### Added

- `Matrix` and `Vector` with shape-checked construction, element access, row
  and column swaps, transpose, `map` and a compensated `trace`.
- `MatrixView` for borrowed rows, columns and sub-blocks, with `submatrix` and
  `set_block` on `Matrix`.
- Arithmetic as `try_add`, `try_sub`, `try_mul`, `try_mul_vec`, `hadamard` and
  `scale`, and through the `+`, `-`, `*` and unary `-` operators.
- Vector norms (1, 2, infinity) and matrix norms (1, infinity, Frobenius,
  maximum).
- Compensated summation (`sum::CompensatedSum`, `sum::sum`, `sum::dot`) and a
  scaled Euclidean norm.
- LU decomposition with partial pivoting: solve, transpose solve, several
  right-hand sides, determinant, inverse and singularity detection with a
  relative tolerance.
- Cholesky decomposition with solve, inverse, determinant and its logarithm.
- Householder QR with a least-squares solve that reports dependent columns.
- A cyclic Jacobi eigenvalue solver for symmetric matrices and power iteration
  for a dominant eigenpair.
- Numerical rank by tolerance, the 1-norm condition number and a Hager-style
  estimate of it.
- Sparse matrices in coordinate and compressed-sparse-row form, with conversion
  between them and to dense, transpose and matrix-vector products.
- Jacobi, Gauss-Seidel and conjugate-gradient solvers with a tolerance, an
  iteration limit, an optional initial guess and a typed non-convergence error.
- Text I/O: a whitespace/CSV format for dense matrices and vectors, the Matrix
  Market coordinate format (`real`, `integer`, `pattern`; `general`,
  `symmetric`, `skew-symmetric`), and file reading with format detection.
  Parse errors carry the line number and the offending field.
- Pretty printing of matrices and vectors with a configurable precision, in
  fixed or scientific notation.
- The `matrixlab` command-line tool with the commands `solve`, `det`, `inv`,
  `eig`, `lstsq` and `info`, and exit codes that separate failed computations,
  usage errors and unreadable input.
- `examples/poisson.rs` and sample data files under `examples/data/`.
