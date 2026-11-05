use matrixlab::{LinalgError, Matrix, Vector};

fn a() -> Matrix {
    Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap()
}

fn b() -> Matrix {
    Matrix::from_rows(&[[0.0, 1.0], [-1.0, 2.0]]).unwrap()
}

#[test]
fn addition_and_subtraction_are_element_wise() {
    assert_eq!(a().try_add(&b()).unwrap().as_slice(), &[1.0, 3.0, 2.0, 6.0]);
    assert_eq!(a().try_sub(&b()).unwrap().as_slice(), &[1.0, 1.0, 4.0, 2.0]);
}

#[test]
fn hadamard_multiplies_matching_elements() {
    assert_eq!(a().hadamard(&b()).unwrap().as_slice(), &[0.0, 2.0, -3.0, 8.0]);
}

#[test]
fn matrix_product_matches_a_hand_computed_result() {
    // [1 2; 3 4] * [0 1; -1 2] = [-2 5; -4 11]
    assert_eq!(a().try_mul(&b()).unwrap().as_slice(), &[-2.0, 5.0, -4.0, 11.0]);
    // and the product does not commute: [0 1; -1 2] * [1 2; 3 4] = [3 4; 5 6]
    assert_eq!(b().try_mul(&a()).unwrap().as_slice(), &[3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn rectangular_product_has_the_outer_shape() {
    let left = Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    let right = Matrix::from_rows(&[[1.0], [0.0], [-1.0]]).unwrap();
    let product = left.try_mul(&right).unwrap();
    assert_eq!(product.shape(), (2, 1));
    assert_eq!(product.as_slice(), &[-2.0, -2.0]);
}

#[test]
fn product_with_the_identity_is_the_matrix_itself() {
    assert_eq!(a().try_mul(&Matrix::identity(2)).unwrap(), a());
    assert_eq!(Matrix::identity(2).try_mul(&a()).unwrap(), a());
}

#[test]
fn mismatched_shapes_give_typed_errors() {
    let wide = Matrix::zeros(2, 3);
    assert_eq!(
        a().try_add(&wide),
        Err(LinalgError::DimensionMismatch {
            op: "matrix addition",
            left: (2, 2),
            right: (2, 3)
        })
    );
    assert_eq!(
        wide.try_mul(&a()),
        Err(LinalgError::DimensionMismatch {
            op: "matrix product",
            left: (2, 3),
            right: (2, 2)
        })
    );
    assert!(a().try_sub(&wide).is_err());
    assert!(a().hadamard(&wide).is_err());
}

#[test]
fn matrix_times_vector() {
    let v = Vector::from_slice(&[1.0, -1.0]);
    assert_eq!(a().try_mul_vec(&v).unwrap().as_slice(), &[-1.0, -1.0]);
    assert_eq!(
        a().try_mul_vec(&Vector::zeros(3)),
        Err(LinalgError::DimensionMismatch {
            op: "matrix-vector product",
            left: (2, 2),
            right: (3, 1)
        })
    );
}

#[test]
fn operators_work_on_owned_and_borrowed_operands() {
    let sum = &a() + &b();
    assert_eq!(sum, a() + b());
    assert_eq!(sum, &a() + b());
    assert_eq!(sum, a() + &b());
    assert_eq!(&a() - &b(), a().try_sub(&b()).unwrap());
    assert_eq!(&a() * &b(), a().try_mul(&b()).unwrap());
}

#[test]
fn scalar_operators_and_negation() {
    assert_eq!((&a() * 2.0).as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    assert_eq!(2.0 * &a(), a() * 2.0);
    assert_eq!((-a()).as_slice(), &[-1.0, -2.0, -3.0, -4.0]);
    let v = Vector::from_slice(&[1.0, -2.0]);
    assert_eq!((-&v).as_slice(), &[-1.0, 2.0]);
    assert_eq!((0.5 * &v).as_slice(), &[0.5, -1.0]);
}

#[test]
fn vector_operators() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    let w = Vector::from_slice(&[3.0, 5.0]);
    assert_eq!((&v + &w).as_slice(), &[4.0, 7.0]);
    assert_eq!((&w - &v).as_slice(), &[2.0, 3.0]);
    assert_eq!((&a() * &v).as_slice(), &[5.0, 11.0]);
}

#[test]
#[should_panic(expected = "dimension mismatch in matrix addition: left is 2x2, right is 3x3")]
fn the_plus_operator_panics_with_the_typed_message() {
    let _ = a() + Matrix::zeros(3, 3);
}

#[test]
#[should_panic(expected = "dimension mismatch in matrix product")]
fn the_star_operator_panics_on_incompatible_shapes() {
    let _ = Matrix::zeros(2, 3) * Matrix::zeros(2, 3);
}
