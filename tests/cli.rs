mod common;

use std::process::Command;

use common::scratch_file;
use matrixlab::cli::{run, EXIT_FAILURE, EXIT_INPUT, EXIT_OK, EXIT_USAGE};

const A: &str = "4 1 0\n1 3 1\n0 1 2\n";
const A_MTX: &str = "%%MatrixMarket matrix coordinate real symmetric
3 3 5
1 1 4
2 1 1
2 2 3
3 2 1
3 3 2
";
const B: &str = "1\n2\n3\n";

/// The outcome of one in-process run.
struct Outcome {
    code: i32,
    stdout: String,
    stderr: String,
}

fn matrixlab(args: &[&str]) -> Outcome {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let code = run(args, &mut stdout, &mut stderr);
    Outcome {
        code,
        stdout: String::from_utf8(stdout).unwrap(),
        stderr: String::from_utf8(stderr).unwrap(),
    }
}

fn file(name: &str, content: &str) -> String {
    scratch_file(name, content).to_str().unwrap().to_string()
}

fn numbers(text: &str) -> Vec<f64> {
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

#[test]
fn solve_prints_one_value_per_line() {
    // det(A) = 18 and the solution is (4, 2, 26) / 18.
    let a = file("cli_solve_a.txt", A);
    let b = file("cli_solve_b.txt", B);
    let outcome = matrixlab(&["solve", &a, &b]);
    assert_eq!(outcome.code, EXIT_OK);
    assert_eq!(outcome.stdout, "0.222222\n0.111111\n1.444444\n");
    assert_eq!(outcome.stderr, "");
}

#[test]
fn every_solve_method_agrees_on_a_matrix_market_system() {
    let a = file("cli_methods_a.mtx", A_MTX);
    let b = file("cli_methods_b.txt", B);
    for method in ["lu", "cholesky", "jacobi", "gauss-seidel", "cg"] {
        let outcome = matrixlab(&["solve", &a, &b, "--method", method, "--precision", "8"]);
        assert_eq!(outcome.code, EXIT_OK, "{}: {}", method, outcome.stderr);
        let x = numbers(&outcome.stdout);
        let expected = [4.0 / 18.0, 2.0 / 18.0, 26.0 / 18.0];
        assert_eq!(x.len(), 3);
        for (found, wanted) in x.iter().zip(expected) {
            assert!((found - wanted).abs() < 1e-7, "{}: {} vs {}", method, found, wanted);
        }
        let iterative = !matches!(method, "lu" | "cholesky");
        assert_eq!(outcome.stderr.starts_with("converged in "), iterative, "{}", method);
    }
}

#[test]
fn an_iteration_limit_that_is_too_small_fails_with_exit_code_one() {
    let a = file("cli_limit_a.txt", A);
    let b = file("cli_limit_b.txt", B);
    let outcome = matrixlab(&["solve", &a, &b, "--method", "jacobi", "--max-iter", "2"]);
    assert_eq!(outcome.code, EXIT_FAILURE);
    assert_eq!(outcome.stdout, "");
    assert!(
        outcome.stderr.starts_with("matrixlab: Jacobi iteration did not converge after 2"),
        "{}",
        outcome.stderr
    );
}

#[test]
fn a_looser_tolerance_needs_fewer_iterations() {
    let a = file("cli_tol_a.txt", A);
    let b = file("cli_tol_b.txt", B);
    let iterations = |tol: &str| -> usize {
        let outcome = matrixlab(&["solve", &a, &b, "--method", "gauss-seidel", "--tol", tol]);
        assert_eq!(outcome.code, EXIT_OK);
        let words: Vec<&str> = outcome.stderr.split_whitespace().collect();
        words[2].parse().unwrap()
    };
    assert!(iterations("1e-3") < iterations("1e-12"));
}

#[test]
fn det_honours_precision_and_scientific_notation() {
    let a = file("cli_det_a.txt", A);
    assert_eq!(matrixlab(&["det", &a]).stdout, "18.000000\n");
    assert_eq!(matrixlab(&["det", &a, "--precision", "2"]).stdout, "18.00\n");
    assert_eq!(matrixlab(&["--precision", "0", "det", &a]).stdout, "18\n");
    assert_eq!(
        matrixlab(&["det", &a, "--scientific", "--precision", "3"]).stdout,
        "1.800e1\n"
    );
}

#[test]
fn det_of_a_singular_matrix_is_zero_and_succeeds() {
    let a = file("cli_det_singular.txt", "1 2\n2 4\n");
    let outcome = matrixlab(&["det", &a]);
    assert_eq!(outcome.code, EXIT_OK);
    assert_eq!(outcome.stdout, "0.000000\n");
}

#[test]
fn inv_output_can_be_fed_back_in() {
    let a = file("cli_inv_a.txt", A);
    let first = matrixlab(&["inv", &a, "--precision", "15"]);
    assert_eq!(first.code, EXIT_OK);
    let inverse = file("cli_inv_inverse.txt", &first.stdout);
    let second = matrixlab(&["inv", &inverse, "--precision", "9"]);
    assert_eq!(second.code, EXIT_OK);
    let back = numbers(&second.stdout);
    let original = numbers(A);
    assert_eq!(back.len(), 9);
    for (found, wanted) in back.iter().zip(original) {
        assert!((found - wanted).abs() < 1e-9, "{} vs {}", found, wanted);
    }
}

#[test]
fn inv_of_a_singular_matrix_fails() {
    let a = file("cli_inv_singular.txt", "1 2\n2 4\n");
    let outcome = matrixlab(&["inv", &a]);
    assert_eq!(outcome.code, EXIT_FAILURE);
    assert_eq!(outcome.stdout, "");
    assert_eq!(
        outcome.stderr,
        "matrixlab: matrix is singular to working precision (pivot 1)\n"
    );
}

#[test]
fn eig_lists_eigenvalues_in_ascending_order() {
    // The eigenvalues of A are 3 and 3 +- sqrt(3).
    let a = file("cli_eig_a.mtx", A_MTX);
    let outcome = matrixlab(&["eig", &a]);
    assert_eq!(outcome.code, EXIT_OK);
    assert_eq!(outcome.stdout, "1.267949\n3.000000\n4.732051\n");
}

#[test]
fn eig_with_vectors_appends_the_eigenvector_matrix() {
    let a = file("cli_eig_vectors.txt", "2 1\n1 2\n");
    let outcome = matrixlab(&["eig", &a, "--vectors", "--precision", "4"]);
    assert_eq!(outcome.code, EXIT_OK);
    let blocks: Vec<&str> = outcome.stdout.split("\n\n").collect();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0], "1.0000\n3.0000");
    let vectors = numbers(blocks[1]);
    assert_eq!(vectors.len(), 4);
    for value in vectors {
        assert!((value.abs() - 0.7071).abs() < 1e-4);
    }
}

#[test]
fn eig_rejects_a_matrix_that_is_not_symmetric() {
    let a = file("cli_eig_asymmetric.txt", "1 2\n3 4\n");
    let outcome = matrixlab(&["eig", &a]);
    assert_eq!(outcome.code, EXIT_FAILURE);
    assert!(outcome.stderr.contains("not symmetric"), "{}", outcome.stderr);
}

#[test]
fn lstsq_fits_a_line_and_reports_the_residual() {
    // Fitting y = c0 + c1 t to (0, 1), (1, 3), (2, 4) gives c = (7/6, 3/2) and a residual
    // of norm 1/sqrt(6).
    let a = file("cli_lstsq_a.csv", "1,0\n1,1\n1,2\n");
    let b = file("cli_lstsq_b.txt", "1 3 4\n");
    let outcome = matrixlab(&["lstsq", &a, &b]);
    assert_eq!(outcome.code, EXIT_OK);
    assert_eq!(outcome.stdout, "1.166667\n1.500000\n");
    assert_eq!(outcome.stderr, "residual norm: 0.408248\n");
}

#[test]
fn lstsq_reports_dependent_columns() {
    let a = file("cli_lstsq_deficient.txt", "1 2\n2 4\n3 6\n");
    let b = file("cli_lstsq_deficient_b.txt", B);
    let outcome = matrixlab(&["lstsq", &a, &b]);
    assert_eq!(outcome.code, EXIT_FAILURE);
    assert!(outcome.stderr.contains("rank deficient"), "{}", outcome.stderr);
}

#[test]
fn info_describes_a_square_sparse_matrix() {
    let a = file("cli_info_a.mtx", A_MTX);
    let outcome = matrixlab(&["info", &a, "--precision", "3"]);
    assert_eq!(outcome.code, EXIT_OK);
    let lines: Vec<&str> = outcome.stdout.lines().collect();
    assert_eq!(
        &lines[..9],
        [
            "shape: 3 x 3",
            "storage: sparse (7 entries)",
            "rank: 3",
            "norm 1: 5.000",
            "norm inf: 5.000",
            "norm frobenius: 5.745",
            "symmetric: yes",
            "trace: 9.000",
            "determinant: 18.000",
        ]
    );
    // ||A||_1 = 5 and ||A^-1||_1 = 16/18.
    let condition: f64 = lines[9]
        .strip_prefix("condition number (1-norm): ")
        .unwrap()
        .parse()
        .unwrap();
    assert!((condition - 40.0 / 9.0).abs() < 1e-12);
    assert_eq!(lines.len(), 10);
}

#[test]
fn info_of_a_singular_matrix_reports_an_infinite_condition_number() {
    let a = file("cli_info_singular.txt", "1 2\n2 4\n");
    let outcome = matrixlab(&["info", &a]);
    assert_eq!(outcome.code, EXIT_OK);
    assert!(outcome.stdout.contains("rank: 1\n"));
    assert!(outcome.stdout.contains("condition number (1-norm): infinite (singular)\n"));
}

#[test]
fn info_of_a_rectangular_matrix_skips_the_square_only_lines() {
    let a = file("cli_info_tall.txt", "1 0\n1 1\n1 2\n");
    let outcome = matrixlab(&["info", &a]);
    assert_eq!(outcome.code, EXIT_OK);
    assert_eq!(outcome.stdout.lines().count(), 6);
    assert!(outcome.stdout.starts_with("shape: 3 x 2\nstorage: dense\nrank: 2\n"));
    assert!(!outcome.stdout.contains("determinant"));
}

#[test]
fn a_right_hand_side_of_the_wrong_length_is_a_failure() {
    let a = file("cli_mismatch_a.txt", A);
    let b = file("cli_mismatch_b.txt", "1\n2\n");
    let outcome = matrixlab(&["solve", &a, &b]);
    assert_eq!(outcome.code, EXIT_FAILURE);
    assert!(outcome.stderr.contains("dimension mismatch"), "{}", outcome.stderr);
}

#[test]
fn unreadable_and_malformed_files_exit_with_code_three() {
    let missing = matrixlab(&["det", "target/matrixlab-no-such-file.txt"]);
    assert_eq!(missing.code, EXIT_INPUT);
    assert!(missing.stderr.starts_with("matrixlab: cannot read target/matrixlab-no-such-file.txt"));

    let bad = file("cli_bad.txt", "1 2\n3 four\n");
    let outcome = matrixlab(&["det", &bad]);
    assert_eq!(outcome.code, EXIT_INPUT);
    assert!(
        outcome.stderr.ends_with("cli_bad.txt: line 2: field 2 is not a number: 'four'\n"),
        "{}",
        outcome.stderr
    );

    let a = file("cli_bad_vector_a.txt", A);
    let matrix_as_vector = matrixlab(&["solve", &a, &a]);
    assert_eq!(matrix_as_vector.code, EXIT_INPUT);
    assert!(matrix_as_vector.stderr.contains("expected a single row or a single column"));
}

#[test]
fn usage_errors_exit_with_code_two() {
    let a = file("cli_usage_a.txt", A);
    let cases: [(&[&str], &str); 13] = [
        (&[], "no command given"),
        (&["frobnicate", &a], "unknown command 'frobnicate'"),
        (&["det"], "'det' takes 1 file argument, got 0"),
        (&["solve", &a], "'solve' takes 2 file arguments, got 1"),
        (&["det", &a, "--bogus"], "unknown option '--bogus'"),
        (&["det", &a, "-x"], "unknown option '-x'"),
        (&["det", &a, "--precision"], "option --precision needs a value"),
        (&["det", &a, "--precision", "many"], "--precision takes a whole number"),
        (&["det", &a, "--precision", "40"], "--precision takes a whole number"),
        (&["solve", &a, &a, "--method", "magic"], "unknown method 'magic'"),
        (&["solve", &a, &a, "--tol", "1e-8"], "need an iterative --method"),
        (&["det", &a, "--method", "lu"], "apply to 'solve' only"),
        (&["det", &a, "--vectors"], "--vectors applies to 'eig' only"),
    ];
    for (args, message) in cases {
        let outcome = matrixlab(args);
        assert_eq!(outcome.code, EXIT_USAGE, "{:?}", args);
        assert_eq!(outcome.stdout, "", "{:?}", args);
        assert!(outcome.stderr.contains(message), "{:?}: {}", args, outcome.stderr);
        assert!(outcome.stderr.contains("matrixlab --help"), "{:?}", args);
    }
}

#[test]
fn bad_iteration_options_are_usage_errors() {
    let a = file("cli_iter_usage_a.txt", A);
    for args in [
        ["--tol", "0"],
        ["--tol", "-1e-3"],
        ["--tol", "nan"],
        ["--max-iter", "0"],
        ["--max-iter", "1.5"],
    ] {
        let outcome = matrixlab(&["solve", &a, &a, "--method", "cg", args[0], args[1]]);
        assert_eq!(outcome.code, EXIT_USAGE, "{:?}", args);
    }
}

#[test]
fn help_and_version_go_to_standard_output() {
    for flag in ["--help", "-h"] {
        let outcome = matrixlab(&[flag]);
        assert_eq!(outcome.code, EXIT_OK);
        assert!(outcome.stdout.starts_with("Usage: matrixlab <command>"));
        for command in ["solve", "det", "inv", "eig", "lstsq", "info"] {
            assert!(outcome.stdout.contains(&format!("\n  {} ", command)), "{}", command);
        }
        assert_eq!(outcome.stderr, "");
    }
    for flag in ["--version", "-V"] {
        let outcome = matrixlab(&[flag]);
        assert_eq!(outcome.code, EXIT_OK);
        assert_eq!(outcome.stdout, format!("matrixlab {}\n", matrixlab::VERSION));
    }
}

#[test]
fn the_binary_reports_results_and_exit_codes() {
    let binary = env!("CARGO_BIN_EXE_matrixlab");
    let a = file("cli_binary_a.txt", A);
    let b = file("cli_binary_b.txt", B);

    let solved = Command::new(binary).args(["solve", &a, &b]).output().unwrap();
    assert_eq!(solved.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&solved.stdout), "0.222222\n0.111111\n1.444444\n");
    assert!(solved.stderr.is_empty());

    let singular = file("cli_binary_singular.txt", "1 2\n2 4\n");
    let failed = Command::new(binary).args(["solve", &singular, &singular]).output().unwrap();
    assert_eq!(failed.status.code(), Some(3));

    let failed = Command::new(binary).args(["inv", &singular]).output().unwrap();
    assert_eq!(failed.status.code(), Some(1));
    assert!(failed.stdout.is_empty());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("singular"));

    let usage = Command::new(binary).output().unwrap();
    assert_eq!(usage.status.code(), Some(2));
}

/// The files in `examples/data` and the outputs shown for them in the README.
#[test]
fn the_example_data_gives_the_documented_results() {
    let data = |name: &str| format!("{}/examples/data/{}", env!("CARGO_MANIFEST_DIR"), name);

    let solved = matrixlab(&["solve", &data("system.txt"), &data("rhs.txt")]);
    assert_eq!(solved.stdout, "0.222222\n0.111111\n1.444444\n");

    let eig = matrixlab(&["eig", &data("laplacian.mtx")]);
    assert_eq!(eig.stdout, "0.267949\n1.000000\n2.000000\n3.000000\n3.732051\n");

    let cg = matrixlab(&[
        "solve",
        &data("laplacian.mtx"),
        &data("ones.txt"),
        "--method",
        "cg",
        "--precision",
        "3",
    ]);
    assert_eq!(cg.code, EXIT_OK);
    assert_eq!(cg.stdout, "2.500\n4.000\n4.500\n4.000\n2.500\n");

    let fit = matrixlab(&["lstsq", &data("line.csv"), &data("observations.txt")]);
    assert_eq!(fit.stdout, "1.090000\n1.940000\n");
    assert_eq!(fit.stderr, "residual norm: 0.286356\n");

    let info = matrixlab(&["info", &data("laplacian.mtx")]);
    assert!(info.stdout.contains("storage: sparse (13 entries)\n"));
    assert!(info.stdout.contains("determinant: 6.000000\n"));

    let not_square = matrixlab(&["inv", &data("line.csv")]);
    assert_eq!(not_square.code, EXIT_FAILURE);
    assert_eq!(
        not_square.stderr,
        "matrixlab: matrix is 4x2, but a square matrix is required\n"
    );
}
