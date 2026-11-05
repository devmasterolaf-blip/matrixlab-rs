# Contributing to matrixlab

Thank you for looking into the code. This file describes how the project is
built and tested and the conventions a change is expected to follow.

## Building and testing

```sh
cargo build
cargo test
```

`cargo test` runs the integration tests in `tests/`, the documentation examples
in `src/` and the Rust examples in `README.md`. All of them must pass, and the
build must be free of compiler warnings, before a change is merged.

The crate has no dependencies, and a new one needs a strong reason: a large part
of the point of the project is that it builds from the standard library alone.

## Layout of a change

- **Library code** goes in the module it belongs to (see the project layout in
  the README). A new decomposition or solver gets its own file under `src/` and
  a line in the module table of `src/lib.rs`.
- **Tests** are integration tests under `tests/`, one file per module, written
  against the public API. Helpers shared by several files are in
  `tests/common/mod.rs`.
- **Documentation**: public types, functions and enum variants have a doc
  comment; for a function it says what is returned and which errors can be
  reported. User-visible changes are recorded in
  `CHANGELOG.md`, and the README is updated when behaviour it describes changes.

## Conventions

### Errors, not panics

Anything that can fail because of its input returns `Result`. Numerical routines
use `LinalgError`; the text parsers use `io::ParseError`, which carries the line
number. Add a variant with the data needed for a useful message rather than
reusing an unrelated one, and give it a `Display` text that reads as a sentence.

Panics are reserved for the places where the standard library panics too:
indexing with `m[(i, j)]`, slicing a row that does not exist, and the arithmetic
operators, which forward to the `try_*` methods and panic with the error's
message.

### Numerical code

- State the algorithm and its source of error in the doc comment or in a comment
  next to the loop that needs it.
- Decide "is this zero?" with an explicit tolerance that is relative to the size
  of the data, and let the caller override it when that is sensible.
- Write comparisons so that a NaN falls on the failing side, for example
  `!(x > threshold)` instead of `x <= threshold`.
- Use `sum::sum` and `sum::dot` where cancellation can matter (residuals, inner
  products of long vectors).

### Tests

- Compare against values that are computed by hand or known analytically, and
  say in a comment where the expected value comes from.
- Every floating-point comparison has an explicit tolerance; exact equality is
  for results that are exact by construction.
- Cover the failure paths: wrong shapes, singular and ill-conditioned input,
  empty input, iteration limits.
- Tests that need files write them with `common::scratch_file`, which uses
  cargo's per-target scratch directory. Do not write to the system temporary
  directory or to the source tree.

### Style

The code is formatted in the default `rustfmt` style (four spaces, lines up to
100 columns, trailing commas in multi-line lists). Run `cargo fmt` if you have
it installed; if you format by hand, match the surrounding code.

### Commits

One self-contained change per commit, with the build and the tests passing.
Messages follow the conventional-commits pattern, for example
`feat(qr): add column pivoting` or `fix(io): reject a size line with a sign`,
followed by a short body that explains why the change is needed.

## Reporting a problem

Please include the smallest input that shows the problem (a matrix of a few
rows is ideal), what you expected, what happened instead, and the output of
`cargo --version`.
