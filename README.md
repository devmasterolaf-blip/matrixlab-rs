# matrixlab

A small dense and sparse linear-algebra library in Rust, with a command-line tool.
It uses the standard library only and is written to be read: every routine is a
direct, commented implementation of a textbook algorithm, and failures are
reported as typed errors rather than panics (the known exceptions are listed under
[Limitations](#limitations)).

This is version 0.1.0, a young project. It is not published on crates.io and it is
not tuned for speed; see [Limitations](#limitations) before relying on it.

## What it does

- **Dense matrices and vectors**: `Matrix` (row-major `f64`), `Vector`, borrowed
  views of rows, columns and sub-blocks, transpose, element-wise and matrix
  arithmetic through `try_*` methods and through operators.
- **Norms**: 1, 2 and infinity norms of vectors; 1, infinity, Frobenius and
  maximum norms of matrices.
- **Decompositions**: LU with partial pivoting (solve, determinant, inverse,
  singularity detection with a tolerance), Cholesky, and Householder QR with a
  least-squares solve.
- **Eigenvalues**: cyclic Jacobi rotations for symmetric matrices, and power
  iteration for a dominant eigenpair.
- **Analysis**: numerical rank by tolerance, the 1-norm condition number, and a
  cheaper estimate of it.
- **Sparse matrices**: coordinate (COO) and compressed sparse row (CSR) forms with
  conversion and matrix-vector products.
- **Iterative solvers**: Jacobi, Gauss-Seidel and conjugate gradient, with a
  relative-residual tolerance, an iteration limit and a typed non-convergence
  error.
- **Careful summation**: compensated (Neumaier) sums and dot products, and a
  scaled Euclidean norm that does not overflow.
- **Text I/O**: a whitespace/CSV format for dense matrices and the Matrix Market
  coordinate format, with parse errors that name the line and the field.
- **Pretty printing** with a configurable precision.
- **A CLI**, `matrixlab`, with the commands `solve`, `det`, `inv`, `eig`, `lstsq`
  and `info`.

## Building

You need a Rust toolchain with `cargo` (edition 2021; developed with Rust 1.93).
There are no dependencies to download.

```sh
cargo build --release
```

The command-line tool is then at `target/release/matrixlab`. To use the library
from another crate, add it as a path or git dependency:

```toml
[dependencies]
matrixlab = { path = "../matrixlab-rs" }
```

## Using the library

Solving a dense system, and what a singular matrix looks like:

```rust
use matrixlab::{LinalgError, Matrix, Vector};

let a = Matrix::from_rows(&[[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]).unwrap();
let b = Vector::from_slice(&[1.0, 2.0, 3.0]);

let x = a.solve(&b).unwrap();
assert!((x[0] - 4.0 / 18.0).abs() < 1e-14);
assert!((a.determinant().unwrap() - 18.0).abs() < 1e-12);

// Keep the decomposition to solve for several right-hand sides.
let lu = a.lu().unwrap();
let y = lu.solve(&Vector::from_slice(&[0.0, 1.0, 0.0])).unwrap();
assert!((&a * &y - Vector::from_slice(&[0.0, 1.0, 0.0])).norm_inf() < 1e-14);

let singular = Matrix::from_rows(&[[1.0, 2.0], [2.0, 4.0]]).unwrap();
assert_eq!(singular.inverse(), Err(LinalgError::Singular { pivot: 1 }));
assert_eq!(singular.rank(), 1);
```

Operators panic on a shape mismatch; the `try_*` methods return the error instead:

```rust
use matrixlab::{LinalgError, Matrix};

let a = Matrix::zeros(2, 3);
let b = Matrix::zeros(2, 2);
let err = a.try_mul(&b).unwrap_err();
assert_eq!(
    err.to_string(),
    "dimension mismatch in matrix product: left is 2x3, right is 2x2"
);
assert!(matches!(err, LinalgError::DimensionMismatch { .. }));
```

Least squares and symmetric eigenvalues:

```rust
use matrixlab::eigen::symmetric_eigen;
use matrixlab::qr::lstsq;
use matrixlab::{Matrix, Vector};

// Fit y = c0 + c1 * t through (0, 1), (1, 3), (2, 4).
let design = Matrix::from_rows(&[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]).unwrap();
let fit = lstsq(&design, &Vector::from_slice(&[1.0, 3.0, 4.0])).unwrap();
assert!((fit.solution[0] - 7.0 / 6.0).abs() < 1e-14);
assert!((fit.solution[1] - 1.5).abs() < 1e-14);

let s = Matrix::from_rows(&[[2.0, 1.0], [1.0, 2.0]]).unwrap();
let eigen = symmetric_eigen(&s).unwrap();
assert!((eigen.values[0] - 1.0).abs() < 1e-14);
assert!((eigen.values[1] - 3.0).abs() < 1e-14);
```

A sparse system solved with conjugate gradient:

```rust
use matrixlab::iterative::{conjugate_gradient, IterativeOptions};
use matrixlab::sparse::CooMatrix;
use matrixlab::Vector;

let mut assembly = CooMatrix::new(3, 3);
for (row, col, value) in [(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0), (2, 2, 2.0)] {
    assembly.push(row, col, value).unwrap();
}
let a = assembly.to_csr();
let b = Vector::from_slice(&[1.0, 2.0, 4.0]);

let result = conjugate_gradient(&a, &b, &IterativeOptions::default()).unwrap();
assert!(result.iterations <= 3);
assert!((result.solution[2] - 2.0).abs() < 1e-10);
```

Reading text, with errors that say where the problem is:

```rust
use matrixlab::io::dense::parse_matrix;
use matrixlab::io::mtx::parse_coordinate;

let m = parse_matrix("1 2\n3 4  # a comment\n").unwrap();
assert_eq!(m.shape(), (2, 2));

let err = parse_matrix("1 2\n3 x\n").unwrap_err();
assert_eq!(err.to_string(), "line 2: field 2 is not a number: 'x'");

let sparse = parse_coordinate(
    "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 4.0\n2 2 5.0\n",
)
.unwrap();
assert_eq!(sparse.nnz(), 2);
```

`examples/poisson.rs` is a complete program that assembles a sparse system and
solves it; run it with `cargo run --example poisson`.

## Using the command-line tool

```text
matrixlab <command> [options] <files>

  solve A b    solve the linear system A x = b
  det A        determinant of a square matrix
  inv A        inverse of a square matrix
  eig A        eigenvalues of a symmetric matrix, in ascending order
  lstsq A b    least-squares solution of an overdetermined system
  info A       shape, norms, rank, determinant and condition number
```

The files used below are in `examples/data/`. After `cargo build --release`, with
`target/release` on your `PATH` (or `cargo run --release --` in place of
`matrixlab`):

```console
$ matrixlab solve examples/data/system.txt examples/data/rhs.txt
0.222222
0.111111
1.444444

$ matrixlab det examples/data/system.txt
18.000000

$ matrixlab inv examples/data/system.txt --precision 4
0.2778 -0.1111 0.0556
-0.1111 0.4444 -0.2222
0.0556 -0.2222 0.6111

$ matrixlab eig examples/data/laplacian.mtx
0.267949
1.000000
2.000000
3.000000
3.732051

$ matrixlab solve examples/data/laplacian.mtx examples/data/ones.txt --method cg --precision 3
converged in 3 iterations, residual norm 0e0
2.500
4.000
4.500
4.000
2.500

$ matrixlab lstsq examples/data/line.csv examples/data/observations.txt
residual norm: 0.286356
1.090000
1.940000

$ matrixlab info examples/data/laplacian.mtx
shape: 5 x 5
storage: sparse (13 entries)
rank: 5
norm 1: 4.000000
norm inf: 4.000000
norm frobenius: 5.291503
symmetric: yes
trace: 10.000000
determinant: 6.000000
condition number (1-norm): 1.8e1
```

Results go to standard output. The remarks of `solve` with an iterative method
(`converged in ...`) and of `lstsq` (`residual norm: ...`) go to standard error,
so the output of `solve`, `det`, `inv` and `lstsq` can be read back as a matrix
or vector file.

Options:

| Option | Meaning |
|---|---|
| `--precision N` | digits after the decimal point, 0 to 17 (default 6) |
| `--scientific` | write numbers in scientific notation |
| `--method M` | `solve` only: `lu` (default), `cholesky`, `jacobi`, `gauss-seidel`, `cg` |
| `--tol T` | relative residual at which an iterative method stops (default 1e-10) |
| `--max-iter N` | iteration limit of an iterative method (default 1000) |
| `--vectors` | `eig` only: also print the eigenvectors, one per column |
| `-h`, `--help` | show the help text |
| `-V`, `--version` | show the version |

Exit codes:

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | the computation failed: singular matrix, shapes that do not fit, no convergence |
| 2 | the command line could not be understood |
| 3 | an input file could not be read or parsed |

```console
$ matrixlab inv examples/data/line.csv
matrixlab: matrix is 4x2, but a square matrix is required
$ echo $?
1
```

## File formats

**Dense text.** One matrix row per line. A line that contains a comma is split at
its commas, any other line at blanks, so both of these are accepted:

```text
# comments start with # or %
1.0  2.0  3.0
4.0, 5.0, 6.0
```

A vector file is a dense file with a single row or a single column.

**Matrix Market.** Files that start with `%%MatrixMarket` are read as coordinate
files: `matrix coordinate` with the field `real`, `integer` or `pattern` and the
symmetry `general`, `symmetric` or `skew-symmetric`. The `array` format,
`complex` values and `hermitian` symmetry are rejected with an error that names
the unsupported word.

## Project layout

```text
src/
  lib.rs         crate documentation and re-exports
  main.rs        the binary: calls cli::run and exits with its code
  cli.rs         argument parsing and the six commands
  error.rs       LinalgError, the error type of the numerical routines
  matrix.rs      Matrix: construction, access, transpose, row and column swaps
  vector.rs      Vector
  view.rs        MatrixView: borrowed rows, columns and blocks
  ops.rs         arithmetic (try_* methods and operator traits)
  norms.rs       matrix norms
  sum.rs         compensated summation, dot product, scaled 2-norm
  lu.rs          LU decomposition: solve, determinant, inverse
  cholesky.rs    Cholesky decomposition
  qr.rs          Householder QR and least squares
  eigen.rs       Jacobi eigenvalue solver and power iteration
  analysis.rs    rank and condition numbers
  sparse/        CooMatrix and CsrMatrix
  iterative.rs   Jacobi, Gauss-Seidel, conjugate gradient
  display.rs     pretty printing
  io/            dense text format, Matrix Market, file reading, ParseError
tests/           integration tests, one file per module plus cli.rs and integration.rs
examples/        poisson.rs and the data files used in this README
```

## Running the tests

```sh
cargo test
```

This runs the integration tests in `tests/`, the documentation examples in the
source and the Rust examples in this README. The expected values in the tests
are computed by hand or known analytically (Hilbert matrices, second-difference
matrices, small integer cases), and every comparison states its tolerance.

## Limitations

- Everything is `f64`; there are no complex numbers and the types are not generic
  over the scalar.
- Dense routines are straightforward loops: no blocking, no SIMD, no threads, no
  BLAS or LAPACK. They are fine for matrices of a few hundred rows.
- Eigenvalues are available for symmetric matrices only (plus power iteration for
  a dominant eigenpair). There is no general eigenvalue solver and no singular
  value decomposition.
- QR has no column pivoting: a rank-deficient least-squares problem is reported
  as an error, not solved in the minimum-norm sense.
- LU decides singularity with a tolerance relative to the largest element of the
  matrix, so a badly scaled but invertible matrix can be reported as singular;
  `Lu::with_tolerance` lets you choose the tolerance.
- The sparse side has no direct solver and no preconditioners.
- The command-line tool converts a sparse matrix to a dense one for every command
  except `solve` with an iterative method. That conversion does not check the
  size first: a file that declares enormous dimensions makes those commands abort
  with an allocation panic instead of an error message.
- Matrix Market support covers the coordinate format only, and the format is
  detected from the very first characters of a file: a blank line or a byte-order
  mark before the `%%MatrixMarket` banner is not handled.
- Non-finite values (`nan`, `inf`) are accepted in input files, and the numerical
  routines do not diagnose them; they surface as NaN results or as unrelated
  errors.

## License

MIT; see [LICENSE](LICENSE).
