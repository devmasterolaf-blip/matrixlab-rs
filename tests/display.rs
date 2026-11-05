use matrixlab::display::{format_matrix, format_number, format_vector, FormatOptions};
use matrixlab::{Matrix, Vector};

fn fixed(precision: usize) -> FormatOptions {
    FormatOptions {
        precision,
        scientific: false,
    }
}

#[test]
fn columns_are_right_aligned_to_their_widest_cell() {
    let m = Matrix::from_rows(&[[1.0, -2.5, 100.0], [10.0, 3.0, 0.5]]).unwrap();
    let text = format_matrix(&m, &fixed(1));
    assert_eq!(text, "[  1.0  -2.5  100.0 ]\n[ 10.0   3.0    0.5 ]");
}

#[test]
fn display_uses_four_decimals_by_default_and_honours_the_precision_flag() {
    let m = Matrix::from_rows(&[[1.0, 0.5]]).unwrap();
    assert_eq!(m.to_string(), "[ 1.0000  0.5000 ]");
    assert_eq!(format!("{:.1}", m), "[ 1.0  0.5 ]");
    assert_eq!(format!("{:.0}", m), "[ 1  0 ]");
    let v = Vector::from_slice(&[1.0, -2.0]);
    assert_eq!(v.to_string(), "[ 1.0000  -2.0000 ]");
    assert_eq!(format!("{:.2}", v), "[ 1.00  -2.00 ]");
}

#[test]
fn scientific_notation() {
    let options = FormatOptions {
        precision: 3,
        scientific: true,
    };
    let v = Vector::from_slice(&[12345.678, 0.00012]);
    assert_eq!(format_vector(&v, &options), "[ 1.235e4  1.200e-4 ]");
}

#[test]
fn values_that_round_to_zero_lose_their_minus_sign() {
    assert_eq!(format_number(-1e-17, &fixed(4)), "0.0000");
    assert_eq!(format_number(-0.0, &fixed(2)), "0.00");
    assert_eq!(format_number(-0.004, &fixed(2)), "0.00");
    assert_eq!(format_number(-0.006, &fixed(2)), "-0.01");
    assert_eq!(format_number(-1.0, &fixed(0)), "-1");
    let sci = FormatOptions {
        precision: 2,
        scientific: true,
    };
    assert_eq!(format_number(-0.0, &sci), "0.00e0");
    assert_eq!(format_number(-1e-17, &sci), "-1.00e-17");
}

#[test]
fn special_values_are_printed_as_rust_prints_them() {
    assert_eq!(format_number(f64::INFINITY, &fixed(2)), "inf");
    assert_eq!(format_number(f64::NEG_INFINITY, &fixed(2)), "-inf");
    assert_eq!(format_number(f64::NAN, &fixed(2)), "NaN");
}

#[test]
fn empty_shapes() {
    assert_eq!(format_matrix(&Matrix::zeros(0, 0), &fixed(2)), "[]");
    assert_eq!(format_matrix(&Matrix::zeros(0, 3), &fixed(2)), "[]");
    assert_eq!(format_matrix(&Matrix::zeros(2, 0), &fixed(2)), "[ ]\n[ ]");
    assert_eq!(format_vector(&Vector::zeros(0), &fixed(2)), "[ ]");
}

#[test]
fn a_column_vector_matrix_prints_one_value_per_line() {
    let m = Matrix::from_rows(&[[1.0], [-22.0], [3.5]]).unwrap();
    assert_eq!(
        format_matrix(&m, &fixed(1)),
        "[   1.0 ]\n[ -22.0 ]\n[   3.5 ]"
    );
}
