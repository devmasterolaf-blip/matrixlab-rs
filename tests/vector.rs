use matrixlab::{LinalgError, Vector};

#[test]
fn constructors_agree_with_each_other() {
    assert_eq!(Vector::zeros(3), Vector::new(vec![0.0, 0.0, 0.0]));
    assert_eq!(Vector::filled(2, 1.5), Vector::from_slice(&[1.5, 1.5]));
    assert_eq!(Vector::unit(3, 1).as_slice(), &[0.0, 1.0, 0.0]);
    let collected: Vector = (1..=3).map(|i| i as f64).collect();
    assert_eq!(collected, Vector::from(vec![1.0, 2.0, 3.0]));
}

#[test]
fn elements_can_be_read_and_written() {
    let mut v = Vector::from_slice(&[1.0, 2.0]);
    v[1] = 5.0;
    assert_eq!(v[1], 5.0);
    assert_eq!(v.get(1), Some(5.0));
    assert_eq!(v.get(2), None);
    assert_eq!(v.len(), 2);
    assert!(!v.is_empty());
    assert_eq!(v.iter().copied().fold(0.0, f64::max), 5.0);
}

#[test]
fn norms_of_a_small_vector() {
    let v = Vector::from_slice(&[3.0, -4.0, 0.0]);
    assert_eq!(v.norm1(), 7.0);
    assert_eq!(v.norm2(), 5.0);
    assert_eq!(v.norm_inf(), 4.0);
}

#[test]
fn norms_of_an_empty_vector_are_zero() {
    let v = Vector::zeros(0);
    assert_eq!(v.norm1(), 0.0);
    assert_eq!(v.norm2(), 0.0);
    assert_eq!(v.norm_inf(), 0.0);
}

#[test]
fn norm_inf_reports_nan() {
    assert!(Vector::from_slice(&[1.0, f64::NAN, 3.0]).norm_inf().is_nan());
}

#[test]
fn dot_product_checks_lengths() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    assert_eq!(a.dot(&b), Ok(32.0));
    let short = Vector::from_slice(&[1.0]);
    assert_eq!(
        a.dot(&short),
        Err(LinalgError::DimensionMismatch {
            op: "dot product",
            left: (3, 1),
            right: (1, 1)
        })
    );
}

#[test]
fn fallible_addition_and_subtraction() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[0.5, -2.0]);
    assert_eq!(a.try_add(&b).unwrap().as_slice(), &[1.5, 0.0]);
    assert_eq!(a.try_sub(&b).unwrap().as_slice(), &[0.5, 4.0]);
    assert!(matches!(
        a.try_add(&Vector::zeros(3)),
        Err(LinalgError::DimensionMismatch {
            op: "vector addition",
            ..
        })
    ));
}

#[test]
fn axpy_updates_in_place() {
    let mut y = Vector::from_slice(&[1.0, 1.0]);
    y.axpy(2.0, &Vector::from_slice(&[3.0, -1.0])).unwrap();
    assert_eq!(y.as_slice(), &[7.0, -1.0]);
    assert!(y.axpy(1.0, &Vector::zeros(1)).is_err());
}

#[test]
fn scale_and_normalized() {
    let v = Vector::from_slice(&[3.0, 4.0]);
    assert_eq!(v.scale(2.0).as_slice(), &[6.0, 8.0]);
    let u = v.normalized().unwrap();
    assert!((u[0] - 0.6).abs() < 1e-15);
    assert!((u[1] - 0.8).abs() < 1e-15);
}

#[test]
fn normalizing_a_zero_or_empty_vector_is_an_error() {
    assert!(matches!(
        Vector::zeros(2).normalized(),
        Err(LinalgError::InvalidArgument { .. })
    ));
    assert_eq!(
        Vector::zeros(0).normalized(),
        Err(LinalgError::Empty { op: "normalization" })
    );
}
