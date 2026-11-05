use matrixlab::LinalgError;

#[test]
fn shape_error_reports_the_expected_element_count() {
    let err = LinalgError::InvalidShape {
        rows: 2,
        cols: 3,
        len: 5,
    };
    assert_eq!(
        err.to_string(),
        "a 2x3 matrix needs 6 elements, but 5 were given"
    );
}

#[test]
fn dimension_mismatch_names_the_operation_and_both_shapes() {
    let err = LinalgError::DimensionMismatch {
        op: "matrix product",
        left: (2, 3),
        right: (2, 2),
    };
    assert_eq!(
        err.to_string(),
        "dimension mismatch in matrix product: left is 2x3, right is 2x2"
    );
}

#[test]
fn no_convergence_prints_the_residual_in_scientific_notation() {
    let err = LinalgError::NoConvergence {
        method: "jacobi",
        iterations: 50,
        residual: 0.00125,
    };
    assert_eq!(
        err.to_string(),
        "jacobi did not converge after 50 iterations (residual 1.25e-3)"
    );
}

#[test]
fn errors_are_usable_as_trait_objects() {
    let boxed: Box<dyn std::error::Error> = Box::new(LinalgError::Singular { pivot: 2 });
    assert_eq!(
        boxed.to_string(),
        "matrix is singular to working precision (pivot 2)"
    );
}
