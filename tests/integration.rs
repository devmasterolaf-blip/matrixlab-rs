//! Tests that run several modules together and check them against each other and against
//! analytically known results.

mod common;

use common::{assert_close, assert_matrix_close, assert_vector_close, hilbert};
use matrixlab::eigen::{power_iteration, symmetric_eigen, PowerOptions};
use matrixlab::io::dense::{parse_matrix, write_matrix, Delimiter};
use matrixlab::io::mtx::{parse_coordinate, write_coordinate};
use matrixlab::iterative::{conjugate_gradient, gauss_seidel, jacobi, IterativeOptions};
use matrixlab::qr::lstsq;
use matrixlab::sparse::CooMatrix;
use matrixlab::{LinalgError, Matrix, Vector};

/// The second-difference matrix with 2 on the diagonal and -1 next to it.
fn laplacian(n: usize) -> CooMatrix {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0).unwrap();
        if i + 1 < n {
            coo.push(i, i + 1, -1.0).unwrap();
            coo.push(i + 1, i, -1.0).unwrap();
        }
    }
    coo
}

/// The solution of `laplacian(n) * x = (1, ..., 1)`: the parabola `x_i = i (n + 1 - i) / 2`
/// with `i` counted from one.
fn parabola(n: usize) -> Vec<f64> {
    (1..=n).map(|i| (i * (n + 1 - i)) as f64 / 2.0).collect()
}

#[test]
fn a_matrix_market_system_solves_the_same_with_every_method() {
    let n = 10;
    let text = write_coordinate(&laplacian(n));
    let coo = parse_coordinate(&text).unwrap();
    let csr = coo.to_csr();
    let dense = coo.to_dense();
    let b = Vector::filled(n, 1.0);
    let expected = parabola(n);
    let options = IterativeOptions {
        tolerance: 1e-12,
        max_iterations: 2000,
        initial_guess: None,
    };

    assert_vector_close(&dense.solve(&b).unwrap(), &expected, 1e-12);
    assert_vector_close(&dense.cholesky().unwrap().solve(&b).unwrap(), &expected, 1e-12);
    let cg = conjugate_gradient(&csr, &b, &options).unwrap();
    assert!(cg.iterations <= n);
    assert_vector_close(&cg.solution, &expected, 1e-10);
    // The stationary methods stop at a relative residual of 1e-12; the error can be larger
    // than the residual by the condition number, about 50 here.
    assert_vector_close(&gauss_seidel(&csr, &b, &options).unwrap().solution, &expected, 1e-9);
    assert_vector_close(&jacobi(&csr, &b, &options).unwrap().solution, &expected, 1e-9);
}

#[test]
fn symmetric_storage_describes_the_same_matrix_as_general_storage() {
    let n = 6;
    let mut text = format!(
        "%%MatrixMarket matrix coordinate real symmetric\n{} {} {}\n",
        n,
        n,
        2 * n - 1
    );
    for i in 1..=n {
        text.push_str(&format!("{} {} 2\n", i, i));
        if i < n {
            text.push_str(&format!("{} {} -1\n", i + 1, i));
        }
    }
    let symmetric = parse_coordinate(&text).unwrap();
    assert_eq!(symmetric.to_csr(), laplacian(n).to_csr());
}

#[test]
fn sparse_and_dense_products_agree_when_entries_repeat() {
    let coo = CooMatrix::from_triplets(
        3,
        3,
        &[(0, 0, 1.5), (2, 1, -2.0), (0, 0, 0.5), (1, 2, 4.0), (2, 1, 1.0)],
    )
    .unwrap();
    let x = Vector::from_slice(&[1.0, -2.0, 3.0]);
    let expected = [2.0, 12.0, 2.0];
    assert_vector_close(&coo.mul_vec(&x).unwrap(), &expected, 0.0);
    assert_vector_close(&coo.to_csr().mul_vec(&x).unwrap(), &expected, 0.0);
    assert_vector_close(&coo.to_dense().try_mul_vec(&x).unwrap(), &expected, 0.0);
}

#[test]
fn a_dense_text_round_trip_does_not_change_a_single_bit_of_the_solution() {
    let a = hilbert(8);
    let b = Vector::filled(8, 1.0);
    let reread = parse_matrix(&write_matrix(&a, Delimiter::Comma)).unwrap();
    assert_eq!(reread, a);
    assert_eq!(reread.solve(&b).unwrap(), a.solve(&b).unwrap());
}

#[test]
fn least_squares_agrees_with_cholesky_on_the_normal_equations() {
    // A cubic through eight points; the basis is tame enough for the normal equations.
    let ts: [f64; 8] = [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0];
    let a = Matrix::from_fn(8, 4, |i, j| ts[i].powi(j as i32));
    let b: Vector = ts.iter().map(|t| (2.0 * t).sin()).collect();
    let fit = lstsq(&a, &b).unwrap();

    let at = a.transpose();
    let normal = at.try_mul(&a).unwrap();
    let rhs = at.try_mul_vec(&b).unwrap();
    let by_cholesky = normal.cholesky().unwrap().solve(&rhs).unwrap();
    assert_vector_close(&fit.solution, by_cholesky.as_slice(), 1e-12);

    // The residual of a least-squares solution is orthogonal to every column.
    let residual = b.try_sub(&a.try_mul_vec(&fit.solution).unwrap()).unwrap();
    assert_close(residual.norm2(), fit.residual_norm, 1e-13);
    assert!(at.try_mul_vec(&residual).unwrap().norm_inf() < 1e-13);
}

#[test]
fn the_spectral_decomposition_rebuilds_the_inverse() {
    let a = laplacian(7).to_dense();
    let eigen = symmetric_eigen(&a).unwrap();
    let reciprocal: Vec<f64> = eigen.values.iter().map(|lambda| 1.0 / lambda).collect();
    let v = &eigen.vectors;
    let rebuilt = v
        .try_mul(&Matrix::from_diagonal(&reciprocal))
        .unwrap()
        .try_mul(&v.transpose())
        .unwrap();
    assert_matrix_close(&rebuilt, &a.inverse().unwrap(), 1e-12);
    // det = n + 1 for this matrix, and it is the product of the eigenvalues.
    assert_close(eigen.values.iter().product::<f64>(), 8.0, 1e-12);
    assert_close(a.determinant().unwrap(), 8.0, 1e-12);
}

#[test]
fn the_spectral_norm_from_power_iteration_sits_between_the_norm_bounds() {
    let a = Matrix::from_rows(&[[4.0, -2.0, 1.0], [3.0, 6.0, -4.0], [2.0, 1.0, 8.0]]).unwrap();
    let gram = a.transpose().try_mul(&a).unwrap();
    let dominant = power_iteration(&gram, &PowerOptions::default()).unwrap();
    let largest = *symmetric_eigen(&gram).unwrap().values.last().unwrap();
    assert_close(dominant.value, largest, 1e-8);

    let spectral = largest.sqrt();
    let frobenius = a.norm_frobenius();
    assert!(spectral <= frobenius);
    assert!(spectral >= frobenius / 3.0_f64.sqrt());
    assert!(spectral <= (a.norm_one() * a.norm_inf()).sqrt());
}

#[test]
fn the_determinant_of_a_block_matrix_factors_through_the_schur_complement() {
    let m = Matrix::from_rows(&[
        [4.0, 1.0, 2.0, 0.0],
        [1.0, 3.0, 0.0, 1.0],
        [2.0, 0.0, 5.0, 2.0],
        [0.0, 1.0, 2.0, 6.0],
    ])
    .unwrap();
    let a = m.submatrix(0..2, 0..2).unwrap();
    let b = m.submatrix(0..2, 2..4).unwrap();
    let c = m.view(2..4, 0..2).unwrap().to_matrix();
    let d = m.view(2..4, 2..4).unwrap().to_matrix();

    let a_inverse_b = a.lu().unwrap().solve_matrix(&b).unwrap();
    let schur = d.try_sub(&c.try_mul(&a_inverse_b).unwrap()).unwrap();
    let product = a.determinant().unwrap() * schur.determinant().unwrap();
    assert_close(m.determinant().unwrap(), product, 1e-10);

    // Putting the blocks back gives the matrix again.
    let mut rebuilt = Matrix::zeros(4, 4);
    rebuilt.set_block(0, 0, &a).unwrap();
    rebuilt.set_block(0, 2, &b).unwrap();
    rebuilt.set_block(2, 0, &c).unwrap();
    rebuilt.set_block(2, 2, &d).unwrap();
    assert_eq!(rebuilt, m);
}

#[test]
fn the_condition_number_bounds_the_effect_of_a_perturbed_right_hand_side() {
    // || dx || / || x || <= cond(A) * || db || / || b || in the 1-norm.
    let a = hilbert(6);
    let b = Vector::filled(6, 1.0);
    let mut perturbed = b.clone();
    perturbed[5] += 1e-8;
    let x = a.solve(&b).unwrap();
    let dx = a.solve(&perturbed).unwrap().try_sub(&x).unwrap();
    let amplification = (dx.norm1() / x.norm1()) / (1e-8 / b.norm1());
    let condition = a.condition_number().unwrap();
    assert!(amplification > 1e3, "amplification {:e}", amplification);
    assert!(amplification <= condition, "{:e} > {:e}", amplification, condition);
}

#[test]
fn a_rank_deficient_matrix_is_reported_consistently() {
    // Row 3 = 2 * row 2 - row 1.
    let a = Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(a.rank(), 2);
    assert!(a.lu().unwrap().is_singular());
    assert_close(a.determinant().unwrap(), 0.0, 1e-12);
    assert!(matches!(a.inverse(), Err(LinalgError::Singular { pivot: 2 })));
    assert!(matches!(a.solve(&b), Err(LinalgError::Singular { pivot: 2 })));
    assert_eq!(a.condition_number(), Ok(f64::INFINITY));
    assert_eq!(a.condition_estimate(), Ok(f64::INFINITY));
    assert!(matches!(lstsq(&a, &b), Err(LinalgError::RankDeficient { column: 2 })));

    // The Gram matrix is symmetric with the eigenvalue zero.
    let gram = a.transpose().try_mul(&a).unwrap();
    let smallest = symmetric_eigen(&gram).unwrap().values[0];
    assert_close(smallest, 0.0, 1e-12);
}
