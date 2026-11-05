//! The `matrixlab` command-line tool.
//!
//! [`run`] is the whole program: it takes the arguments and two writers and returns the
//! exit code, so it can be driven from tests without starting a process.

use std::io::Write;

use crate::display::{format_number, FormatOptions};
use crate::eigen::symmetric_eigen;
use crate::error::LinalgError;
use crate::io::{read_matrix, read_vector, MatrixData, ReadError};
use crate::iterative::{
    conjugate_gradient, gauss_seidel, jacobi, IterativeOptions, IterativeSolution,
};
use crate::matrix::Matrix;
use crate::qr::lstsq;
use crate::vector::Vector;

/// Exit code of a successful run.
pub const EXIT_OK: i32 = 0;
/// Exit code when the computation fails: a singular matrix, shapes that do not fit, an
/// iteration that does not converge.
pub const EXIT_FAILURE: i32 = 1;
/// Exit code for a command line that cannot be understood.
pub const EXIT_USAGE: i32 = 2;
/// Exit code when an input file cannot be read or parsed.
pub const EXIT_INPUT: i32 = 3;

/// The number of digits after the decimal point when `--precision` is not given.
pub const DEFAULT_PRECISION: usize = 6;

const USAGE: &str = "\
Usage: matrixlab <command> [options] <files>

Commands:
  solve A b    solve the linear system A x = b
  det A        determinant of a square matrix
  inv A        inverse of a square matrix
  eig A        eigenvalues of a symmetric matrix, in ascending order
  lstsq A b    least-squares solution of an overdetermined system
  info A       shape, norms, rank, determinant and condition number

Matrix files are plain text (one row per line, fields separated by blanks or
commas, # or % comments) or Matrix Market coordinate files; the format is
detected from the content. Vector files are plain text.

Options:
  --precision N   digits after the decimal point (default 6)
  --scientific    write numbers in scientific notation
  --method M      solve only: lu (default), cholesky, jacobi, gauss-seidel, cg
  --tol T         relative residual at which an iterative method stops
  --max-iter N    iteration limit of an iterative method
  --vectors       eig only: also print the eigenvectors, one per column
  -h, --help      show this text
  -V, --version   show the version

Exit codes: 0 success, 1 the computation failed, 2 usage error,
3 an input file could not be read or parsed.
";

#[derive(Debug)]
enum CliError {
    Usage(String),
    Input(ReadError),
    Numerical(LinalgError),
}

impl From<ReadError> for CliError {
    fn from(error: ReadError) -> CliError {
        CliError::Input(error)
    }
}

impl From<LinalgError> for CliError {
    fn from(error: LinalgError) -> CliError {
        CliError::Numerical(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Lu,
    Cholesky,
    Jacobi,
    GaussSeidel,
    ConjugateGradient,
}

impl Method {
    fn is_iterative(self) -> bool {
        matches!(
            self,
            Method::Jacobi | Method::GaussSeidel | Method::ConjugateGradient
        )
    }
}

/// The parsed command line.
#[derive(Debug)]
struct Invocation {
    command: String,
    files: Vec<String>,
    format: FormatOptions,
    method: Option<Method>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
    vectors: bool,
}

/// What a command produces: the result for standard output and remarks for standard error.
#[derive(Debug, Default)]
struct Output {
    result: String,
    notes: String,
}

/// Runs the tool. `args` are the arguments after the program name.
pub fn run<S: AsRef<str>>(args: &[S], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let args: Vec<&str> = args.iter().map(|arg| arg.as_ref()).collect();
    if args.iter().any(|arg| *arg == "-h" || *arg == "--help") {
        return emit(out, USAGE);
    }
    if args.iter().any(|arg| *arg == "-V" || *arg == "--version") {
        return emit(out, &format!("matrixlab {}\n", crate::VERSION));
    }
    match parse_args(&args).and_then(|invocation| execute(&invocation)) {
        Ok(output) => {
            let _ = err.write_all(output.notes.as_bytes());
            emit(out, &output.result)
        }
        Err(CliError::Usage(message)) => {
            let _ = writeln!(err, "matrixlab: {}", message);
            let _ = writeln!(err, "Try 'matrixlab --help' for the list of commands and options.");
            EXIT_USAGE
        }
        Err(CliError::Input(error)) => {
            let _ = writeln!(err, "matrixlab: {}", error);
            EXIT_INPUT
        }
        Err(CliError::Numerical(error)) => {
            let _ = writeln!(err, "matrixlab: {}", error);
            EXIT_FAILURE
        }
    }
}

/// Writes the result; a writer that fails (a closed pipe, a full disk) is a failed run.
fn emit(out: &mut dyn Write, text: &str) -> i32 {
    match out.write_all(text.as_bytes()).and_then(|()| out.flush()) {
        Ok(()) => EXIT_OK,
        Err(_) => EXIT_FAILURE,
    }
}

fn usage<T>(message: String) -> Result<T, CliError> {
    Err(CliError::Usage(message))
}

fn parse_args(args: &[&str]) -> Result<Invocation, CliError> {
    let mut positional = Vec::new();
    let mut precision = DEFAULT_PRECISION;
    let mut scientific = false;
    let mut method = None;
    let mut tolerance = None;
    let mut max_iterations = None;
    let mut vectors = false;

    let mut rest = args.iter();
    while let Some(&arg) = rest.next() {
        let mut value = |name: &str| match rest.next() {
            Some(&value) => Ok(value),
            None => usage(format!("option {} needs a value", name)),
        };
        match arg {
            "--precision" => {
                let text = value(arg)?;
                precision = match text.parse::<usize>() {
                    Ok(digits) if digits <= 17 => digits,
                    _ => {
                        return usage(format!(
                            "--precision takes a whole number from 0 to 17, got '{}'",
                            text
                        ))
                    }
                };
            }
            "--scientific" => scientific = true,
            "--method" => method = Some(parse_method(value(arg)?)?),
            "--tol" => {
                let text = value(arg)?;
                tolerance = match text.parse::<f64>() {
                    Ok(tol) if tol > 0.0 && tol.is_finite() => Some(tol),
                    _ => return usage(format!("--tol takes a positive number, got '{}'", text)),
                };
            }
            "--max-iter" => {
                let text = value(arg)?;
                max_iterations = match text.parse::<usize>() {
                    Ok(limit) if limit > 0 => Some(limit),
                    _ => {
                        return usage(format!(
                            "--max-iter takes a positive whole number, got '{}'",
                            text
                        ))
                    }
                };
            }
            "--vectors" => vectors = true,
            _ if arg.len() > 1 && arg.starts_with('-') => {
                return usage(format!("unknown option '{}'", arg))
            }
            _ => positional.push(arg.to_string()),
        }
    }

    let mut positional = positional.into_iter();
    let command = match positional.next() {
        Some(command) => command,
        None => return usage("no command given".to_string()),
    };
    let files: Vec<String> = positional.collect();
    let expected_files = match command.as_str() {
        "solve" | "lstsq" => 2,
        "det" | "inv" | "eig" | "info" => 1,
        _ => return usage(format!("unknown command '{}'", command)),
    };
    if files.len() != expected_files {
        return usage(format!(
            "'{}' takes {} file argument{}, got {}",
            command,
            expected_files,
            if expected_files == 1 { "" } else { "s" },
            files.len()
        ));
    }
    if command != "solve" && (method.is_some() || tolerance.is_some() || max_iterations.is_some())
    {
        return usage("--method, --tol and --max-iter apply to 'solve' only".to_string());
    }
    let iterative = method.map_or(false, Method::is_iterative);
    if !iterative && (tolerance.is_some() || max_iterations.is_some()) {
        return usage("--tol and --max-iter need an iterative --method".to_string());
    }
    if vectors && command != "eig" {
        return usage("--vectors applies to 'eig' only".to_string());
    }

    Ok(Invocation {
        command,
        files,
        format: FormatOptions {
            precision,
            scientific,
        },
        method,
        tolerance,
        max_iterations,
        vectors,
    })
}

fn parse_method(name: &str) -> Result<Method, CliError> {
    match name {
        "lu" => Ok(Method::Lu),
        "cholesky" => Ok(Method::Cholesky),
        "jacobi" => Ok(Method::Jacobi),
        "gauss-seidel" => Ok(Method::GaussSeidel),
        "cg" => Ok(Method::ConjugateGradient),
        _ => usage(format!(
            "unknown method '{}' (expected lu, cholesky, jacobi, gauss-seidel or cg)",
            name
        )),
    }
}

fn execute(invocation: &Invocation) -> Result<Output, CliError> {
    let matrix = read_matrix(&invocation.files[0])?;
    let format = &invocation.format;
    match invocation.command.as_str() {
        "solve" => {
            let b = read_vector(&invocation.files[1])?;
            solve(invocation, &matrix, &b)
        }
        "lstsq" => {
            let b = read_vector(&invocation.files[1])?;
            let fit = lstsq(&matrix.to_dense(), &b)?;
            Ok(Output {
                result: vector_lines(&fit.solution, format),
                notes: format!(
                    "residual norm: {}\n",
                    format_number(fit.residual_norm, format)
                ),
            })
        }
        "det" => {
            let determinant = matrix.to_dense().determinant()?;
            Ok(Output {
                result: format!("{}\n", format_number(determinant, format)),
                notes: String::new(),
            })
        }
        "inv" => {
            let inverse = matrix.to_dense().inverse()?;
            Ok(Output {
                result: matrix_lines(&inverse, format),
                notes: String::new(),
            })
        }
        "eig" => {
            let eigen = symmetric_eigen(&matrix.to_dense())?;
            let mut result = vector_lines(&Vector::new(eigen.values), format);
            if invocation.vectors {
                result.push('\n');
                result.push_str(&matrix_lines(&eigen.vectors, format));
            }
            Ok(Output {
                result,
                notes: String::new(),
            })
        }
        _ => info(&matrix, format),
    }
}

fn solve(invocation: &Invocation, matrix: &MatrixData, b: &Vector) -> Result<Output, CliError> {
    let format = &invocation.format;
    let method = invocation.method.unwrap_or(Method::Lu);
    if !method.is_iterative() {
        let dense = matrix.to_dense();
        let x = if method == Method::Cholesky {
            dense.cholesky()?.solve(b)?
        } else {
            dense.solve(b)?
        };
        return Ok(Output {
            result: vector_lines(&x, format),
            notes: String::new(),
        });
    }

    let mut options = IterativeOptions::default();
    if let Some(tolerance) = invocation.tolerance {
        options.tolerance = tolerance;
    }
    if let Some(limit) = invocation.max_iterations {
        options.max_iterations = limit;
    }
    let csr = matrix.to_csr();
    let IterativeSolution {
        solution,
        iterations,
        residual_norm,
    } = match method {
        Method::Jacobi => jacobi(&csr, b, &options)?,
        Method::GaussSeidel => gauss_seidel(&csr, b, &options)?,
        _ => conjugate_gradient(&csr, b, &options)?,
    };
    Ok(Output {
        result: vector_lines(&solution, format),
        notes: format!(
            "converged in {} iterations, residual norm {:e}\n",
            iterations, residual_norm
        ),
    })
}

fn info(matrix: &MatrixData, format: &FormatOptions) -> Result<Output, CliError> {
    let dense = matrix.to_dense();
    let number = |value: f64| format_number(value, format);
    let mut lines = vec![
        format!("shape: {} x {}", dense.rows(), dense.cols()),
        match matrix {
            MatrixData::Dense(_) => "storage: dense".to_string(),
            MatrixData::Sparse(coo) => format!("storage: sparse ({} entries)", coo.nnz()),
        },
        format!("rank: {}", dense.rank()),
        format!("norm 1: {}", number(dense.norm_one())),
        format!("norm inf: {}", number(dense.norm_inf())),
        format!("norm frobenius: {}", number(dense.norm_frobenius())),
    ];
    if dense.is_square() && !dense.is_empty() {
        let symmetric = dense.is_symmetric(1e-12 * dense.norm_max());
        lines.push(format!("symmetric: {}", if symmetric { "yes" } else { "no" }));
        lines.push(format!("trace: {}", number(dense.trace()?)));
        lines.push(format!("determinant: {}", number(dense.determinant()?)));
        let condition = dense.condition_number()?;
        lines.push(if condition.is_finite() {
            format!("condition number (1-norm): {:e}", condition)
        } else {
            "condition number (1-norm): infinite (singular)".to_string()
        });
    }
    let mut result = lines.join("\n");
    result.push('\n');
    Ok(Output {
        result,
        notes: String::new(),
    })
}

fn vector_lines(vector: &Vector, format: &FormatOptions) -> String {
    let mut text = String::new();
    for &value in vector.iter() {
        text.push_str(&format_number(value, format));
        text.push('\n');
    }
    text
}

fn matrix_lines(matrix: &Matrix, format: &FormatOptions) -> String {
    let mut text = String::new();
    for row in matrix.iter_rows() {
        let fields: Vec<String> = row.iter().map(|&value| format_number(value, format)).collect();
        text.push_str(&fields.join(" "));
        text.push('\n');
    }
    text
}
