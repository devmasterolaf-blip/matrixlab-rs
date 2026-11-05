mod common;

use common::{assert_close, assert_matrix_close, hilbert};
use matrixlab::eigen::{symmetric_eigen, symmetric_eigen_with, JacobiOptions};
use matrixlab::{LinalgError, Matrix, Vector};

#[test]
fn two_by_two_has_eigenvalues_one_and_three() {
    let a = Matrix::from_rows(&[[2.0, 1.0], [1.0, 2.0]]).unwrap();
    let eig = symmetric_eigen(&a).unwrap();
    assert_close(eig.values[0], 1.0, 1e-14);
    assert_close(eig.values[1], 3.0, 1e-14);
    let h = std::f64::consts::FRAC_1_SQRT_2;
    // Eigenvectors (1, -1) / sqrt(2) and (1, 1) / sqrt(2), largest component positive.
    assert_close(eig.vectors[(0, 0)].abs(), h, 1e-14);
    assert_close(eig.vectors[(0, 0)] * eig.vectors[(1, 0)], -0.5, 1e-14);
    assert_close(eig.vectors[(0, 1)], h, 1e-14);
    assert_close(eig.vectors[(1, 1)], h, 1e-14);
}

#[test]
fn tridiagonal_matrix_has_the_known_spectrum() {
    // Eigenvalues of tridiag(1, 4, 1) of order 3 are 4 + 2 cos(k pi / 4), k = 1, 2, 3.
    let a = Matrix::from_rows(&[[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]]).unwrap();
    let eig = symmetric_eigen(&a).unwrap();
    let root2 = std::f64::consts::SQRT_2;
    assert_close(eig.values[0], 4.0 - root2, 1e-13);
    assert_close(eig.values[1], 4.0, 1e-13);
    assert_close(eig.values[2], 4.0 + root2, 1e-13);
}

#[test]
fn discrete_laplacian_spectrum() {
    let n = 8;
    let a = Matrix::from_fn(n, n, |i, j| {
        if i == j {
            2.0
        } else if i.abs_diff(j) == 1 {
            -1.0
        } else {
            0.0
        }
    });
    let eig = symmetric_eigen(&a).unwrap();
    for (k, &value) in eig.values.iter().enumerate() {
        let angle = (k + 1) as f64 * std::f64::consts::PI / (n + 1) as f64;
        assert_close(value, 2.0 - 2.0 * angle.cos(), 1e-13);
    }
}

#[test]
fn a_diagonal_matrix_needs_no_sweeps_and_is_sorted() {
    let a = Matrix::from_diagonal(&[3.0, -1.0, 2.0]);
    let eig = symmetric_eigen(&a).unwrap();
    assert_eq!(eig.values, vec![-1.0, 2.0, 3.0]);
    assert_eq!(eig.sweeps, 0);
    // The eigenvector of -1 is the second unit vector.
    assert_eq!(eig.vectors.column(0), vec![0.0, 1.0, 0.0]);
}

#[test]
fn eigenvectors_are_orthonormal_and_diagonalise_the_matrix() {
    let a = hilbert(6);
    let eig = symmetric_eigen(&a).unwrap();
    let v = &eig.vectors;
    assert_matrix_close(&v.transpose().try_mul(v).unwrap(), &Matrix::identity(6), 1e-13);
    let reconstructed = v
        .try_mul(&Matrix::from_diagonal(&eig.values))
        .unwrap()
        .try_mul(&v.transpose())
        .unwrap();
    assert_matrix_close(&reconstructed, &a, 1e-13);
}

#[test]
fn each_pair_satisfies_the_eigen_equation() {
    let a = Matrix::from_rows(&[
        [4.0, -30.0, 60.0, -35.0],
        [-30.0, 300.0, -675.0, 420.0],
        [60.0, -675.0, 1620.0, -1050.0],
        [-35.0, 420.0, -1050.0, 700.0],
    ])
    .unwrap();
    let eig = symmetric_eigen(&a).unwrap();
    for (k, &lambda) in eig.values.iter().enumerate() {
        let v = Vector::new(eig.vectors.column(k));
        let residual = a.try_mul_vec(&v).unwrap().try_sub(&v.scale(lambda)).unwrap();
        assert_close(residual.norm2(), 0.0, 1e-11 * a.norm_frobenius());
    }
}

#[test]
fn trace_and_determinant_match_the_spectrum() {
    let a = hilbert(4);
    let eig = symmetric_eigen(&a).unwrap();
    let sum: f64 = eig.values.iter().sum();
    let product: f64 = eig.values.iter().product();
    assert_close(sum, a.trace().unwrap(), 1e-14);
    // det(H4) = 1 / 6048000
    assert_close(product, 1.0 / 6_048_000.0, 1e-18);
}

#[test]
fn repeated_eigenvalues_still_give_an_orthonormal_basis() {
    let a = Matrix::from_rows(&[[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]]).unwrap();
    // Eigenvalues 1, 3, 3.
    let eig = symmetric_eigen(&a).unwrap();
    assert_close(eig.values[0], 1.0, 1e-14);
    assert_close(eig.values[1], 3.0, 1e-14);
    assert_close(eig.values[2], 3.0, 1e-14);
    let v = &eig.vectors;
    assert_matrix_close(&v.transpose().try_mul(v).unwrap(), &Matrix::identity(3), 1e-14);
}

#[test]
fn a_one_by_one_matrix_is_its_own_eigenvalue() {
    let eig = symmetric_eigen(&Matrix::from_rows(&[[-7.5]]).unwrap()).unwrap();
    assert_eq!(eig.values, vec![-7.5]);
    assert_eq!(eig.vectors[(0, 0)], 1.0);
}

#[test]
fn the_sweep_limit_gives_a_typed_error() {
    let options = JacobiOptions {
        max_sweeps: 1,
        tolerance: 1e-15,
    };
    let err = symmetric_eigen_with(&hilbert(6), &options).unwrap_err();
    match err {
        LinalgError::NoConvergence {
            method,
            iterations,
            residual,
        } => {
            assert_eq!(method, "Jacobi eigenvalue iteration");
            assert_eq!(iterations, 1);
            assert!(residual > 0.0 && residual < 1.0);
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let asymmetric = Matrix::from_rows(&[[1.0, 2.0], [0.0, 1.0]]).unwrap();
    assert_eq!(
        symmetric_eigen(&asymmetric).unwrap_err(),
        LinalgError::NotSymmetric { row: 0, col: 1 }
    );
    assert_eq!(
        symmetric_eigen(&Matrix::zeros(2, 3)).unwrap_err(),
        LinalgError::NotSquare { rows: 2, cols: 3 }
    );
    assert!(matches!(
        symmetric_eigen(&Matrix::zeros(0, 0)).unwrap_err(),
        LinalgError::Empty { .. }
    ));
    let options = JacobiOptions {
        max_sweeps: 10,
        tolerance: -1.0,
    };
    assert!(matches!(
        symmetric_eigen_with(&Matrix::identity(2), &options).unwrap_err(),
        LinalgError::InvalidArgument { .. }
    ));
}

mod power {
    use super::common::{assert_close, assert_vector_close};
    use matrixlab::eigen::{power_iteration, symmetric_eigen, PowerOptions};
    use matrixlab::{LinalgError, Matrix, Vector};

    #[test]
    fn finds_the_dominant_eigenvalue_of_a_symmetric_matrix() {
        let a = Matrix::from_rows(&[[2.0, 1.0], [1.0, 2.0]]).unwrap();
        let pair = power_iteration(&a, &PowerOptions::default()).unwrap();
        assert_close(pair.value, 3.0, 1e-9);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert_vector_close(&pair.vector, &[h, h], 1e-9);
        assert!(pair.iterations > 1);
    }

    #[test]
    fn finds_a_negative_dominant_eigenvalue() {
        // Eigenvalues -5 and 1: the vector alternates in sign but the quotient settles.
        let a = Matrix::from_rows(&[[-2.0, 3.0], [3.0, -2.0]]).unwrap();
        let pair = power_iteration(&a, &PowerOptions::default()).unwrap();
        assert_close(pair.value, -5.0, 1e-9);
    }

    #[test]
    fn works_for_a_non_symmetric_matrix_with_a_real_dominant_eigenvalue() {
        // Upper triangular, so the eigenvalues are the diagonal: 5 and 2.
        let a = Matrix::from_rows(&[[5.0, 1.0], [0.0, 2.0]]).unwrap();
        let pair = power_iteration(&a, &PowerOptions::default()).unwrap();
        assert_close(pair.value, 5.0, 1e-8);
        assert_close(pair.vector[1], 0.0, 1e-8);
    }

    #[test]
    fn agrees_with_the_jacobi_solver_on_a_larger_matrix() {
        let a = Matrix::from_fn(6, 6, |i, j| 1.0 / (i + j + 1) as f64);
        let largest = *symmetric_eigen(&a).unwrap().values.last().unwrap();
        let pair = power_iteration(&a, &PowerOptions::default()).unwrap();
        assert_close(pair.value, largest, 1e-9);
    }

    #[test]
    fn an_exact_eigenvector_as_start_converges_at_once() {
        let a = Matrix::from_diagonal(&[1.0, 4.0]);
        let options = PowerOptions {
            start: Some(Vector::from_slice(&[0.0, 2.0])),
            ..PowerOptions::default()
        };
        let pair = power_iteration(&a, &options).unwrap();
        assert_eq!(pair.iterations, 1);
        assert_eq!(pair.value, 4.0);
        assert_eq!(pair.vector.as_slice(), &[0.0, 1.0]);
    }

    #[test]
    fn eigenvalues_of_equal_magnitude_do_not_converge() {
        let a = Matrix::from_diagonal(&[2.0, -2.0]);
        let options = PowerOptions {
            max_iterations: 200,
            ..PowerOptions::default()
        };
        match power_iteration(&a, &options).unwrap_err() {
            LinalgError::NoConvergence {
                method,
                iterations,
                residual,
            } => {
                assert_eq!(method, "power iteration");
                assert_eq!(iterations, 200);
                assert!(residual > 0.1);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn a_close_second_eigenvalue_needs_more_iterations_than_allowed() {
        let a = Matrix::from_diagonal(&[1.0, 0.999]);
        let options = PowerOptions {
            max_iterations: 20,
            ..PowerOptions::default()
        };
        assert!(matches!(
            power_iteration(&a, &options),
            Err(LinalgError::NoConvergence { iterations: 20, .. })
        ));
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let a = Matrix::identity(2);
        let wrong_length = PowerOptions {
            start: Some(Vector::zeros(3)),
            ..PowerOptions::default()
        };
        assert!(matches!(
            power_iteration(&a, &wrong_length),
            Err(LinalgError::DimensionMismatch { .. })
        ));
        let zero_start = PowerOptions {
            start: Some(Vector::zeros(2)),
            ..PowerOptions::default()
        };
        assert!(matches!(
            power_iteration(&a, &zero_start),
            Err(LinalgError::InvalidArgument { .. })
        ));
        assert!(matches!(
            power_iteration(&Matrix::zeros(2, 3), &PowerOptions::default()),
            Err(LinalgError::NotSquare { .. })
        ));
        assert!(matches!(
            power_iteration(&Matrix::zeros(0, 0), &PowerOptions::default()),
            Err(LinalgError::Empty { .. })
        ));
    }
}
