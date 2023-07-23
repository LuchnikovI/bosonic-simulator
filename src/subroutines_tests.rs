use num_complex::{ComplexFloat, Complex64};
use num_traits::One;
use rand::{thread_rng, Rng};


use crate::subroutines::*;
use crate::subroutines_utils::{get_size, Term, Op};
use crate::test_utils::*;

fn _test_apply_term<const N: usize>(
    all_encodings: &[usize],
    positions: [usize; N],
    op_types: [Op; N],
)
{
    let size = 2usize.pow(get_size(all_encodings) as u32);
    let term = Term {
        positions,
        op_types,
    };
    let delta = Complex64::new(0.3, 0.7);
    let mut rng = thread_rng();
    let src: Vec<_> = (0..size).map(|_| Complex64::new(rng.gen(), rng.gen())).collect();
    let mut dst: Vec<_> = (0..size).map(|_| Complex64::new(rng.gen(), rng.gen())).collect();
    let src_clone = src.clone();
    let mut dst_clone = dst.clone();
    apply_term(&mut dst, &src, &term, &all_encodings, delta);
    apply_term_test(&mut dst_clone, &src_clone, &term, &all_encodings, delta);
    for (i, (v1, v2)) in dst.into_iter().zip(dst_clone).enumerate() {
        assert!((v1 - v2).abs() < 1e-10, "Iter. number: {}, lhs: {}, rhs: {}", i, v1, v2);
    }
}

#[test]
fn test_apply_term()
{
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0], [Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1], [Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [2], [Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [3], [Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [4], [Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [5], [Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0], [Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1], [Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [2], [Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [3], [Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [4], [Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [5], [Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0], [Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1], [Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [2], [Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [3], [Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [4], [Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [5], [Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0], [Op::N2]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1], [Op::N2]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [2], [Op::N2]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [3], [Op::N2]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [4], [Op::N2]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [5], [Op::N2]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1], [Op::Rising, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 2], [Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 3], [Op::Rising, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 5], [Op::Rising, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1, 3], [Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [2, 5], [Op::Rising, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [3, 4], [Op::Rising, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [4, 5], [Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1], [Op::N, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 2], [Op::Lowering, Op::N2]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 3], [Op::Rising, Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 5], [Op::N2, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1, 3], [Op::Lowering, Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 2], [Op::Rising, Op::Lowering, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 2, 3], [Op::Rising, Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 3, 5], [Op::Rising, Op::Rising, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 5], [Op::Rising, Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1, 3, 4], [Op::Lowering, Op::Lowering, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [2, 3, 5], [Op::Rising, Op::Rising, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1, 3, 4], [Op::Rising, Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 4, 5], [Op::Lowering, Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 2], [Op::Rising, Op::Lowering, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 2, 3], [Op::N, Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 3, 5], [Op::Rising, Op::N, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 5], [Op::Rising, Op::Lowering, Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1, 3, 4], [Op::N, Op::N2, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [2, 3, 5], [Op::Rising, Op::N2, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1, 3, 4], [Op::N, Op::Lowering, Op::N2]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 4, 5], [Op::N, Op::N2, Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 2, 3], [Op::Rising, Op::Lowering, Op::Rising, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1, 2, 3, 4], [Op::Rising, Op::Lowering, Op::Lowering, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 2, 3, 5], [Op::Rising, Op::Rising, Op::Rising, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 3, 5], [Op::Rising, Op::Lowering, Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 4, 5], [Op::Lowering, Op::Lowering, Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 2, 3], [Op::N, Op::Lowering, Op::Rising, Op::N]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [1, 2, 3, 4], [Op::Rising, Op::N2, Op::Lowering, Op::N2]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 2, 3, 5], [Op::Rising, Op::Rising, Op::Rising, Op::Rising]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 3, 5], [Op::N, Op::N, Op::Lowering, Op::Lowering]);
    _test_apply_term(&[2, 1, 2, 3, 2, 1], [0, 1, 4, 5], [Op::N, Op::Lowering, Op::N2, Op::Lowering]);
}

#[test]
fn test_init_std()
{
    let all_encodings = [3, 2, 1, 2, 3, 2, 1];
    let size = 2usize.pow(get_size(&all_encodings) as u32);
    let state = init_std::<Complex64>(&all_encodings);
    assert_eq!(size, state.len());
    for elem in &state[1..] {
        assert!(elem.abs() < 1e-10);
    }
    assert!((state[0] - Complex64::one()).abs() < 1e-10);
}


fn _test_get_density<const N: usize>(
    all_encodings: &[usize],
    positions: [usize; N],
)
{
    let size = 2usize.pow(get_size(all_encodings) as u32);
    let mut rng = thread_rng();
    let src: Vec<_> = (0..size).map(|_| Complex64::new(rng.gen(), rng.gen())).collect();
    let test_dens = get_density_test(&src, &positions, &all_encodings);
    let dens = get_density(&src, &positions, &all_encodings);
    for (d1, d2) in dens.into_iter().zip(test_dens).skip(200)
    {
        assert!((d1 - d2).abs() < 1e-10);
    }
}

#[test]
fn test_get_density()
{
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [1]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [2]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [3]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [4]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [1]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [2]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [3]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [4]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 1]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 2]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 3]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [1, 3]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [2, 5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [3, 4]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [4, 5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 1, 2]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 2, 3]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 3, 5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 1, 5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [1, 3, 4]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [2, 3, 5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [1, 3, 4]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 4, 5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 1, 2, 3]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [1, 2, 3, 4]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 2, 3, 5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 1, 3, 5]);
    _test_get_density(&[2, 1, 2, 3, 2, 1], [0, 1, 4, 5]);
}