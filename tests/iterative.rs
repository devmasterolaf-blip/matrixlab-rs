mod common;

use common::{assert_close, assert_vector_close};
use matrixlab::iterative::{conjugate_gradient, gauss_seidel, jacobi, IterativeOptions};
use matrixlab::sparse::{CooMatrix, CsrMatrix};
use matrixlab::{LinalgError, Matrix, Vector};

/// tridiag(-1, 4, -1) of order 3: strictly diagonally dominant and positive definite.
fn dominant() -> CsrMatrix {
    CsrMatrix::from_dense(
        &Matrix::from_rows(&[[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]]).unwrap(),
    )
}

/// The right-hand side for which the solution of `dominant()` is (1, 2, 3).
fn rhs() -> Vector {
    Vector::from_slice(&[2.0, 4.0, 10.0])
}

/// The second-difference matrix tridiag(-1, 2, -1) of order `n`.
fn laplacian(n: usize) -> CsrMatrix {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0).unwrap();
        if i + 1 < n {
            coo.push(i, i + 1, -1.0).unwrap();
            coo.push(i + 1, i, -1.0).unwrap();
        }
    }
    coo.to_csr()
}

fn tight() -> IterativeOptions {
    IterativeOptions {
        tolerance: 1e-13,
        ..IterativeOptions::default()
    }
}

#[test]
fn jacobi_converges_on_a_diagonally_dominant_system() {
    let result = jacobi(&dominant(), &rhs(), &tight()).unwrap();
    assert_vector_close(&result.solution, &[1.0, 2.0, 3.0], 1e-12);
    assert!(result.residual_norm <= 1e-13 * rhs().norm2());
    assert!(result.iterations > 5);
}

#[test]
fn gauss_seidel_converges_in_fewer_sweeps_than_jacobi() {
    let slow = jacobi(&dominant(), &rhs(), &tight()).unwrap();
    let fast = gauss_seidel(&dominant(), &rhs(), &tight()).unwrap();
    assert_vector_close(&fast.solution, &[1.0, 2.0, 3.0], 1e-12);
    assert!(
        fast.iterations < slow.iterations,
        "Gauss-Seidel took {} sweeps, Jacobi {}",
        fast.iterations,
        slow.iterations
    );
}

#[test]
fn conjugate_gradient_solves_the_same_system_in_at_most_n_steps() {
    let result = conjugate_gradient(&dominant(), &rhs(), &tight()).unwrap();
    assert_vector_close(&result.solution, &[1.0, 2.0, 3.0], 1e-12);
    assert!(result.iterations <= 3);
}

#[test]
fn conjugate_gradient_matches_a_direct_solve_on_the_laplacian() {
    let n = 30;
    let a = laplacian(n);
    let b = Vector::filled(n, 1.0);
    let result = conjugate_gradient(&a, &b, &tight()).unwrap();
    let direct = a.to_dense().solve(&b).unwrap();
    assert_vector_close(&result.solution, direct.as_slice(), 1e-9);
    // With a constant right-hand side the solution is the parabola x_i = (i + 1)(n - i) / 2.
    assert_close(result.solution[0], n as f64 / 2.0, 1e-9);
    assert!(result.iterations <= n);
}

#[test]
fn gauss_seidel_converges_on_a_matrix_that_is_only_positive_definite() {
    // The Laplacian is not strictly diagonally dominant, but it is symmetric positive
    // definite, which is enough for Gauss-Seidel.
    let a = laplacian(6);
    let x_true = Vector::from_slice(&[1.0, -1.0, 2.0, 0.0, 3.0, 1.0]);
    let b = a.mul_vec(&x_true).unwrap();
    let options = IterativeOptions {
        max_iterations: 5000,
        tolerance: 1e-12,
        initial_guess: None,
    };
    let result = gauss_seidel(&a, &b, &options).unwrap();
    assert_vector_close(&result.solution, x_true.as_slice(), 1e-10);
}

#[test]
fn jacobi_reports_divergence_as_no_convergence() {
    // The iteration matrix has spectral radius sqrt(6), so the error grows every sweep.
    let a = CsrMatrix::from_dense(&Matrix::from_rows(&[[1.0, 2.0], [3.0, 1.0]]).unwrap());
    let options = IterativeOptions {
        max_iterations: 40,
        ..IterativeOptions::default()
    };
    match jacobi(&a, &Vector::from_slice(&[1.0, 1.0]), &options).unwrap_err() {
        LinalgError::NoConvergence {
            method,
            iterations,
            residual,
        } => {
            assert_eq!(method, "Jacobi iteration");
            assert_eq!(iterations, 40);
            assert!(residual > 1e6);
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn divergence_to_infinity_stops_before_the_iteration_limit() {
    let a = CsrMatrix::from_dense(&Matrix::from_rows(&[[1.0, 1e200], [1e200, 1.0]]).unwrap());
    let options = IterativeOptions {
        max_iterations: 1_000_000,
        ..IterativeOptions::default()
    };
    match jacobi(&a, &Vector::from_slice(&[1.0, 1.0]), &options).unwrap_err() {
        LinalgError::NoConvergence { iterations, .. } => assert!(iterations < 10),
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn the_iteration_limit_is_respected() {
    let a = laplacian(50);
    let b = Vector::filled(50, 1.0);
    let options = IterativeOptions {
        max_iterations: 5,
        ..IterativeOptions::default()
    };
    for result in [
        jacobi(&a, &b, &options),
        gauss_seidel(&a, &b, &options),
        conjugate_gradient(&a, &b, &options),
    ] {
        match result.unwrap_err() {
            LinalgError::NoConvergence {
                iterations,
                residual,
                ..
            } => {
                assert_eq!(iterations, 5);
                assert!(residual > 0.0);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }
}

#[test]
fn an_exact_initial_guess_needs_no_iterations() {
    let options = IterativeOptions {
        initial_guess: Some(Vector::from_slice(&[1.0, 2.0, 3.0])),
        ..IterativeOptions::default()
    };
    for result in [
        jacobi(&dominant(), &rhs(), &options),
        gauss_seidel(&dominant(), &rhs(), &options),
        conjugate_gradient(&dominant(), &rhs(), &options),
    ] {
        let result = result.unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.residual_norm, 0.0);
    }
}

#[test]
fn a_good_initial_guess_saves_iterations() {
    let cold = gauss_seidel(&dominant(), &rhs(), &tight()).unwrap();
    let options = IterativeOptions {
        initial_guess: Some(Vector::from_slice(&[1.0, 2.0, 3.000001])),
        ..tight()
    };
    let warm = gauss_seidel(&dominant(), &rhs(), &options).unwrap();
    assert!(warm.iterations < cold.iterations);
}

#[test]
fn a_zero_right_hand_side_gives_the_zero_solution() {
    let options = IterativeOptions {
        initial_guess: Some(Vector::from_slice(&[5.0, -5.0, 5.0])),
        ..IterativeOptions::default()
    };
    for result in [
        jacobi(&dominant(), &Vector::zeros(3), &options),
        gauss_seidel(&dominant(), &Vector::zeros(3), &options),
        conjugate_gradient(&dominant(), &Vector::zeros(3), &options),
    ] {
        let result = result.unwrap();
        assert_eq!(result.solution, Vector::zeros(3));
        assert_eq!(result.iterations, 0);
    }
}

#[test]
fn stationary_methods_need_a_non_zero_diagonal() {
    let a = CsrMatrix::from_dense(&Matrix::from_rows(&[[2.0, 1.0], [1.0, 0.0]]).unwrap());
    let b = Vector::from_slice(&[1.0, 1.0]);
    assert_eq!(
        jacobi(&a, &b, &IterativeOptions::default()).unwrap_err(),
        LinalgError::ZeroDiagonal { row: 1 }
    );
    assert_eq!(
        gauss_seidel(&a, &b, &IterativeOptions::default()).unwrap_err(),
        LinalgError::ZeroDiagonal { row: 1 }
    );
}

#[test]
fn conjugate_gradient_rejects_an_indefinite_matrix() {
    let a = CsrMatrix::from_dense(&Matrix::from_diagonal(&[1.0, -1.0]));
    let err = conjugate_gradient(
        &a,
        &Vector::from_slice(&[1.0, 1.0]),
        &IterativeOptions::default(),
    )
    .unwrap_err();
    assert_eq!(err, LinalgError::NotPositiveDefinite { pivot: 0 });
}

#[test]
fn conjugate_gradient_rejects_an_asymmetric_matrix() {
    let a = CsrMatrix::from_dense(&Matrix::from_rows(&[[4.0, 1.0], [0.0, 3.0]]).unwrap());
    let err = conjugate_gradient(
        &a,
        &Vector::from_slice(&[1.0, 1.0]),
        &IterativeOptions::default(),
    )
    .unwrap_err();
    assert_eq!(err, LinalgError::NotSymmetric { row: 0, col: 1 });
}

#[test]
fn shapes_and_options_are_validated() {
    let b = rhs();
    let defaults = IterativeOptions::default();
    let rectangular = CooMatrix::new(2, 3).to_csr();
    assert_eq!(
        jacobi(&rectangular, &b, &defaults).unwrap_err(),
        LinalgError::NotSquare { rows: 2, cols: 3 }
    );
    assert_eq!(
        gauss_seidel(&dominant(), &Vector::zeros(2), &defaults).unwrap_err(),
        LinalgError::DimensionMismatch {
            op: "Gauss-Seidel iteration",
            left: (3, 3),
            right: (2, 1)
        }
    );
    let bad_guess = IterativeOptions {
        initial_guess: Some(Vector::zeros(4)),
        ..IterativeOptions::default()
    };
    assert!(matches!(
        conjugate_gradient(&dominant(), &b, &bad_guess).unwrap_err(),
        LinalgError::DimensionMismatch {
            op: "initial guess",
            ..
        }
    ));
    let bad_tolerance = IterativeOptions {
        tolerance: f64::NAN,
        ..IterativeOptions::default()
    };
    assert!(matches!(
        jacobi(&dominant(), &b, &bad_tolerance).unwrap_err(),
        LinalgError::InvalidArgument { .. }
    ));
    assert!(matches!(
        conjugate_gradient(&CooMatrix::new(0, 0).to_csr(), &Vector::zeros(0), &defaults)
            .unwrap_err(),
        LinalgError::Empty { .. }
    ));
}
