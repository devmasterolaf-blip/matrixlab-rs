use matrixlab::sum::{dot, norm2, sum, CompensatedSum};

#[test]
fn compensated_sum_keeps_small_terms_next_to_a_large_one() {
    // The naive left-to-right sum of these values is 0.0: both ones are absorbed by 1e100.
    let values = [1.0, 1e100, 1.0, -1e100];
    let naive: f64 = values.iter().sum();
    assert_eq!(naive, 0.0);
    assert_eq!(sum(values.iter().copied()), 2.0);
}

#[test]
fn many_tenths_add_up_to_the_exact_decimal_value() {
    let total = sum(std::iter::repeat(0.1).take(1_000_000));
    assert!((total - 100_000.0).abs() < 1e-9, "total = {}", total);
}

#[test]
fn accumulator_can_be_fed_incrementally() {
    let mut acc = CompensatedSum::new();
    assert_eq!(acc.value(), 0.0);
    for i in 1..=100 {
        acc.add(i as f64);
    }
    assert_eq!(acc.value(), 5050.0);
}

#[test]
fn dot_product_survives_cancellation() {
    let a = [1e16, 1.0, -1e16];
    let b = [1.0, 1.0, 1.0];
    assert_eq!(dot(&a, &b), 1.0);
}

#[test]
fn norm2_matches_the_3_4_5_triangle() {
    assert_eq!(norm2(&[3.0, 4.0]), 5.0);
    assert_eq!(norm2(&[]), 0.0);
    assert_eq!(norm2(&[0.0, 0.0]), 0.0);
}

#[test]
fn norm2_does_not_overflow_or_underflow() {
    let big = norm2(&[3e200, 4e200]);
    assert!((big - 5e200).abs() <= 1e186, "big = {:e}", big);
    let small = norm2(&[3e-200, 4e-200]);
    assert!((small - 5e-200).abs() <= 1e-214, "small = {:e}", small);
}

#[test]
fn norm2_propagates_infinity_and_nan() {
    assert_eq!(norm2(&[1.0, f64::INFINITY]), f64::INFINITY);
    assert!(norm2(&[1.0, f64::NAN]).is_nan());
    assert!(norm2(&[f64::INFINITY, f64::NAN]).is_nan());
}
