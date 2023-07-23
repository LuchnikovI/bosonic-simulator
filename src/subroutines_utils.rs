use std::{ops::Mul, fmt::Debug};
use num_complex::{ComplexFloat, Complex};
use num_traits::{Float, FloatConst, One};
use serde::{Serialize, Deserialize};

pub trait TrueComplex: ComplexFloat
{
    fn new(re: Self::Real, im: Self::Real) -> Self;
}

impl<T: Float + FloatConst> TrueComplex for Complex<T>
{
    fn new(re: Self::Real, im: Self::Real) -> Self {
        Complex::new(re, im)
    }
}

pub trait Value:
    ComplexFloat +
    Serialize +
    for <'de> Deserialize<'de> +
    Debug +
    Clone +
    Send +
    Sync +
    Copy +
    'static
{}

impl<T: 'static> Value for T
where
    T: 
    ComplexFloat +
    Serialize +
    for <'de> Deserialize<'de> +
    Debug +
    Clone +
    Send +
    Sync +
    Copy
{}

#[derive(
    Deserialize,
    Serialize,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Copy
)]
pub enum Op {
    #[serde(rename = "A+")]
    Rising,
    #[serde(rename = "A-")]
    Lowering,
    #[serde(rename = "N1")]
    N,
    #[serde(rename = "N2")]
    N2,
}

#[derive(
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Copy
)]
pub struct Term<const N: usize>
{
    pub positions: [usize; N],
    pub op_types: [Op; N],
}

impl<const N: usize> Term<N> {

    pub(super) fn transpose(&self) -> Self
    {
        let mut transposed_term = self.clone();
        for op in transposed_term.op_types.iter_mut() {
            match op {
                Op::Lowering => *op = Op::Rising,
                Op::Rising => *op = Op::Lowering,
                Op::N => {},
                Op::N2 => {},
            }
        }
        transposed_term
    }

    pub(super) fn sort(&mut self)
    {
        let mut pairs = [(0usize, Op::Lowering); N];
        for ((dst_pos, dst_op_type), (src_pos, src_op_type)) in
        pairs.iter_mut().zip(self.positions.iter().zip(self.op_types.iter()))
        {
            *dst_pos = *src_pos;
            *dst_op_type = *src_op_type;
        }
        pairs.sort();
        for ((dst_pos, dst_op_type), (src_pos, src_op_type)) in
        self.positions.iter_mut().zip(&mut self.op_types).zip(pairs.into_iter())
        {
            *dst_pos = src_pos;
            *dst_op_type = src_op_type;
        }
    }

}

// ----------------------------------------------------------------------------------------

#[inline]
fn get_tensor_product<Operands, Operand, T>(
    operands: &Operands,
) -> Vec<T>
where
    Operands: AsRef<[Operand]>,
    Operand: AsRef<[T]>,
    T: Mul + One + Clone + Copy,
{
    let size = operands.as_ref().iter().map(|x| x.as_ref().len()).product();
    let mut tensor_product = Vec::with_capacity(size);
    for mut i in 0..size {
        let mut value = T::one();
        for operand in operands.as_ref() {
            let j = i % operand.as_ref().len();
            value = value * operand.as_ref()[j];
            i /= operand.as_ref().len();
        }
        tensor_product.push(value);
    }
    tensor_product
}

#[inline]
fn get_diagonal_offset<const N: usize>(
    term: &Term<N>,
    all_encodings: &[usize],
) -> isize
{
    let mut diag_pos = 0isize;
    for (pos, op) in term.positions.into_iter().zip(term.op_types).rev() {
        diag_pos *= 2usize.pow(all_encodings[pos] as u32) as isize;
        match op {
            Op::Rising => diag_pos -= 1,
            Op::Lowering => diag_pos += 1,
            Op::N => {},
            Op::N2 => {},
        };
    }
    diag_pos
}

#[inline]
fn diag_per_operator<T: Value>(
    op: Op,
    size: usize,
) -> Vec<T>
{
    match op {
        Op::Rising => {
            (1..size).chain(0..1).map(|x| T::from(x).unwrap().sqrt()).collect::<Vec<_>>()
        },
        Op::Lowering => {
            (0..size).map(|x| T::from(x).unwrap().sqrt()).collect::<Vec<_>>()
        },
        Op::N => {
            (0..size).map(|x| T::from(x).unwrap()).collect::<Vec<_>>()
        }
        Op::N2 => {
            (0..size).map(|x| T::from(x).unwrap().powi(2)).collect::<Vec<_>>()
        }
    }
}

#[inline]
pub(super) fn get_diagonal<const N: usize, T: Value>(
    term: &Term<N>,
    all_encodings: &[usize],
) -> (Vec<T>, isize)
{
    let offset = get_diagonal_offset(term, all_encodings);
    let mut operands = Vec::with_capacity(N);
    for (pos, op) in term.positions.into_iter().zip(term.op_types) {
        operands.push(diag_per_operator(op, 2usize.pow(all_encodings[pos] as u32)));
    }
    let diagonal = get_tensor_product(&operands);
    (diagonal, offset)
}

// ----------------------------------------------------------------------------------------

#[inline]
pub(super) fn masks_and_offsets<const N: usize>(
    all_encodings: &[usize],
    terms: &[usize; N],
) -> ([usize; N], [usize; N])
{
    let mut shifts = [0; N];
    let mut masks = [0; N];
    all_encodings.into_iter().fold((0, 0, 0), |(mut iter, mut start_full, mut start_reduced), &x|
    {
        for (i, term) in terms.into_iter().enumerate()
        {
            if *term == iter {
                masks[i] = ((1 << x) - 1) << start_full;
                shifts[i] = start_full - start_reduced;
                start_reduced += x;
            }
        }
        start_full += x;
        iter += 1;
        (iter, start_full, start_reduced)
    });
    (masks, shifts)
}

#[inline(always)]
pub(super) fn get_operator_index<const N: usize>(
    index: usize,
    shifts: &[usize; N],
    masks: &[usize; N],
) -> usize
{
    let mut operator_index = (masks[0] & index) >> shifts[0];
    for (mask, shift) in masks.into_iter().zip(shifts.into_iter()).skip(1)
    {
        operator_index |= (mask & index) >> shift;
    }
    operator_index
}

#[inline]
pub(super) fn get_offset<const N: usize>(
    operator_index: usize,
    shifts: &[usize; N],
    masks: &[usize; N],
) -> usize
{
    let mut index = ((masks[0] >> shifts[0]) & operator_index) << shifts[0];
    for (mask, shift) in masks.into_iter().zip(shifts.into_iter()).skip(1)
    {
        index |= ((mask >> shift) & operator_index) << shift;
    }
    index
}

#[inline]
pub(super) fn get_global_offset<const N: usize>(
    term: &Term<N>,
    all_encodings: &[usize],
) -> isize
{
    let mut all_encodings = all_encodings.to_owned();
    all_encodings.reverse();
    let mut diag_pos = 0isize;
    let mut start = 0usize;
    for (pos, op_type) in term.positions.iter().zip(term.op_types).rev()
    {
        let pos = all_encodings.len() - pos - 1;
        let dim = 2usize.pow((&all_encodings[start..=pos]).iter().map(|x| *x).sum::<usize>() as u32);
        diag_pos *= dim as isize;
        match op_type {
            Op::Rising => diag_pos -= 1,
            Op::Lowering => diag_pos += 1,
            Op::N => {},
            Op::N2 => {},
        }
        start = pos + 1;
    }
    if start < all_encodings.len() {
        let dim = 2usize.pow((&all_encodings[start..]).iter().map(|x| *x).sum::<usize>() as u32);
        diag_pos *= dim as isize;
    }
    diag_pos
}

#[inline]
pub(super) fn get_size(
    all_encodings: &[usize]
) -> usize
{
    all_encodings.into_iter().sum()
}

// ---------------------------------------------------------------------------------------

#[inline]
pub(super) fn get_batch_size<const N: usize>(
    all_encodings: &[usize],
    positions: &[usize; N],
) -> usize
{
    all_encodings.into_iter()
        .enumerate()
        .filter(|(encoding_pos, _)| {
            let mut predicate = true;
            for pos in positions {
                if *pos == *encoding_pos {
                    predicate = false;
                }
            }
            predicate
        })
        .map(|(_, val)| 2usize.pow(*val as u32))
        .product()
}

#[inline]
pub(super) fn get_strides<const N: usize>(
    all_encodings: &[usize],
    positions: &[usize; N],
) -> [usize; N]
{
    let mut strides = [0usize; N];
    for (stride, pos) in strides.iter_mut().zip(positions.into_iter()) {
        *stride = all_encodings[..*pos].iter()
            .map(|x| 2usize.pow(*x as u32))
            .product();
    }
    strides
}

#[inline]
pub(super) fn get_target_encodings<const N: usize>(
    all_encodings: &[usize],
    positions: &[usize; N],
) -> [usize; N]
{
    let mut target_encodings = [0; N];
    for (pos, encoding)
    in positions.into_iter().zip(target_encodings.iter_mut())
    {
        *encoding = all_encodings[*pos];
    }
    target_encodings
}

#[inline]
pub(super) fn get_masks<const N: usize>(sorted_strides: &[usize; N]) -> [usize; N]
{
    let mut masks = [0; N];
    for (mask, stride) in masks.iter_mut().zip(sorted_strides.iter())
    {
        *mask = usize::MAX - (stride - 1);
    }
    masks
}

#[inline(always)]
pub(super) fn get_batch_index<const N: usize>(
    mut index: usize,
    sorted_masks: &[usize; N],
    sorted_target_encodings: &[usize; N],
) -> usize
{
    for (mask, encoding) in sorted_masks.into_iter().zip(sorted_target_encodings)
    {
        index = ((mask & index) << encoding) | ((!mask) & index)
    }
    index
}

#[inline]
pub(super) fn get_density_size<const N: usize>(
    all_encodings: &[usize],
    positions: &[usize; N],
) -> usize
{
    2usize.pow(positions.into_iter().map(|index| all_encodings[*index]).sum::<usize>() as u32)
}

#[inline]
pub(super) fn density_matrix_shifts_and_masks<const N: usize>(
    all_encodings: &[usize],
    terms: &[usize; N],
) -> ([usize; N], [usize; N])
{
    let mut shifts = [0; N];
    let mut masks = [0; N];
    all_encodings.into_iter().fold((0, 0, 0), |(mut iter, mut start_full, mut start_reduced), &x|
    {
        for (i, term) in terms.into_iter().enumerate()
        {
            if *term == iter {
                masks[i] = ((1 << x) - 1) << start_reduced;
                shifts[i] = start_full - start_reduced;
                start_reduced += x;
            }
        }
        start_full += x;
        iter += 1;
        (iter, start_full, start_reduced)
    });
    (masks, shifts)
}

#[inline(always)]
pub(super) fn density_matrix_index_to_state_index<const N: usize>(
    index: usize,
    shifts: &[usize; N],
    masks: &[usize; N],
) -> usize
{
    let mut operator_index = (masks[0] & index) << shifts[0];
    for (mask, shift) in masks.into_iter().zip(shifts.into_iter()).skip(1)
    {
        operator_index |= (mask & index) << shift;
    }
    operator_index
}

// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use num_complex::Complex64;
    use super::*;

    #[test]
        fn test_dens_matrix_utils() {
            let all_encodings = [1, 2, 3, 2, 1, 1, 2, 1, 2, 3];
            let positions = [];
            let size = get_batch_size(&all_encodings, &positions);
            assert_eq!(2usize.pow(18), size);
            let strides = get_strides(&all_encodings, &positions);
            let target_encodings = get_target_encodings(&all_encodings, &positions);
            let masks = get_masks(&strides);
            let batch_index = get_batch_index(usize::MAX, &masks, &target_encodings);
            assert_eq!(usize::MAX, batch_index);
            let batch_index = get_batch_index(0usize, &masks, &target_encodings);
            assert_eq!(0, batch_index);
            let batch_index = get_batch_index(512usize - 1, &masks, &target_encodings);
            assert_eq!(512usize - 1, batch_index);
            // --------------------------------------------------------------------------
            let all_encodings = [1, 2, 3, 2, 1, 1, 2, 1, 2, 3];
            let mut positions = [5, 2, 8];
            positions.sort();
            let size = get_batch_size(&all_encodings, &positions);
            assert_eq!(2usize.pow(12), size);
            let strides = get_strides(&all_encodings, &positions);
            assert_eq!([8, 512, 8192], strides);
            let target_encodings = get_target_encodings(&all_encodings, &positions);
            assert_eq!(target_encodings, [3, 1, 2]);
            let masks = get_masks(&strides);
            assert_eq!([usize::MAX << 3, usize::MAX << 9, usize::MAX << 13], masks);
            let batch_index = get_batch_index(usize::MAX, &masks, &target_encodings);
            assert_eq!(0b1111111111111111111111111111111111111111111111111001110111000111, batch_index);
            let batch_index = get_batch_index(0usize, &masks, &target_encodings);
            assert_eq!(0, batch_index);
            let batch_index = get_batch_index(512usize - 1, &masks, &target_encodings);
            assert_eq!(0b1110111000111, batch_index);
            // --------------------------------------------------------------------------
            let all_encodings = [1, 2, 3, 2, 1, 1, 2, 1, 2, 3];
            let mut positions = [7, 0, 3, 9];
            positions.sort();
            let size = get_batch_size(&all_encodings, &positions);
            assert_eq!(2usize.pow(11), size);
            let strides = get_strides(&all_encodings, &positions);
            assert_eq!([1, 64, 4096, 32768], strides);
            let target_encodings = get_target_encodings(&all_encodings, &positions);
            assert_eq!(target_encodings, [1, 2, 1, 3]);
            let masks = get_masks(&strides);
            assert_eq!([usize::MAX, usize::MAX << 6, usize::MAX << 12, usize::MAX << 15], masks);
            let batch_index = get_batch_index(usize::MAX, &masks, &target_encodings);
            assert_eq!(0b1111111111111111111111111111111111111111111111000110111100111110, batch_index);
            let batch_index = get_batch_index(512usize - 1, &masks, &target_encodings);
            assert_eq!(0b111100111110, batch_index);
            let batch_index = get_batch_index(0, &masks, &target_encodings);
            assert_eq!(0, batch_index);
        }

    #[test]
    fn test_indexing()
    {
        let (masks, shifts) = masks_and_offsets(
            &[1, 2, 1, 3, 4, 3, 1, 3],
            &[3],
        );
        let index: usize = 0b000000000000000000000010000;
        let operator_index = get_operator_index(index, &shifts, &masks);
        assert_eq!(1, operator_index);
        let offset = get_offset(operator_index, &shifts, &masks);
        assert_eq!(index, offset);
        // ------------------------------------------------------------
        let (masks, shifts) = masks_and_offsets(
            &[1, 2, 1, 3, 4, 3, 1, 3],
            &[2, 4, 7],
        );
        let index: usize = 0b0000111000011110001000;
        let operator_index = get_operator_index(index, &shifts, &masks);
        assert_eq!(2usize.pow(8) - 1, operator_index);
        let offset = get_offset(operator_index, &shifts, &masks);
        assert_eq!(index, offset);
    }

    #[test]
    fn test_product()
    {
        let tensor_product = get_tensor_product(
            &[
                &[1, 2, 3],
            ]
        );
        assert_eq!(vec![1, 2, 3], tensor_product);
        let tensor_product = get_tensor_product(
            &[
                [1, 2, 3].as_slice(),
                [3, 2].as_slice()
            ]
        );
        assert_eq!(vec![3, 6, 9, 2, 4, 6], tensor_product);
        let tensor_product = get_tensor_product(
            &[
                [2, 1, 4].as_slice(),
                [2, 3].as_slice(),
                [8, 3].as_slice(),
            ]
        );
        assert_eq!(vec![32, 16, 64, 48, 24, 96, 12, 6, 24, 18, 9, 36], tensor_product);
    }

    #[test]
    fn test_get_offset() {
        let term = Term {
            positions: [1, 3, 5],
            op_types: [Op::Lowering, Op::Rising, Op::Lowering]
        };
        let all_encodings = [1, 3, 1, 4, 2, 2, 1];
        let (masks, shifts) = masks_and_offsets(&all_encodings, &term.positions);
        let offset = get_offset(123, &shifts, &masks);
        let index = get_operator_index(offset, &shifts, &masks);
        assert_eq!(123, index);
    }

    fn _test_get_diagonal<const N: usize>(
        term: Term<N>,
        all_encodings: &[usize],
        true_diag: impl Iterator<Item=Complex64>,
    )
    {
        let transposed_term = term.transpose();
        let (diag, offset) = get_diagonal::<N, Complex64>(
            &term,
            &all_encodings,
        );
        let (transposed_diag, transposed_offset) = get_diagonal::<N, Complex64>(
            &transposed_term,
            &all_encodings,
        );
        for (a, b) in true_diag.into_iter().zip(&diag) {
            assert!((a - *b).abs() < 1e-10);
        }
        if offset > 0 {
            for (a, b) in diag.into_iter().skip(offset as usize).zip(transposed_diag) {
                assert!((a - b).abs() < 1e-10);
            }
        } else {
            for (a, b) in diag.into_iter().zip(&transposed_diag[(-offset as usize)..]) {
                assert!((a - b).abs() < 1e-10);
            }
        }
        assert_eq!(offset, -transposed_offset);
    }

    #[test]
    fn test_get_diagonal() {
        let term = Term {
            positions: [1, 3, 5],
            op_types: [Op::Lowering, Op::Rising, Op::Lowering]
        };
        let true_diag = (0..4).map(|a| Complex64::from(a as f64).sqrt())
            .flat_map(|x| {
                (1..16).chain(0..1).map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            })
            .flat_map(|x| {
                (0..8).map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            });
        let all_encodings = [1, 3, 1, 4, 2, 2, 1];
        _test_get_diagonal(term, &all_encodings, true_diag);
        let term = Term {
            positions: [1, 2, 3, 7],
            op_types: [Op::Rising, Op::Lowering, Op::Rising, Op::Lowering]
        };
        let true_diag = (0..2).map(|a| Complex64::from(a as f64).sqrt())
            .flat_map(|x| {
                (1..16).chain(0..1).map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            })
            .flat_map(|x| {
                (0..2).map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            })
            .flat_map(|x| {
                (1..8).chain(0..1).map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            });
        let all_encodings = [1, 3, 1, 4, 2, 2, 1, 1, 2, 3, 4, 3, 1];
        _test_get_diagonal(term, &all_encodings, true_diag);
    }

    #[test]
    fn test_density_matrix_indexing()
    {
        let all_encodings = [1, 4, 1, 3, 2, 1, 5];
        let positions = [1, 3, 6];
        let (masks, shifts) = density_matrix_shifts_and_masks(&all_encodings, &positions);
        let index = 0b100011011001;
        let state_index = density_matrix_index_to_state_index(index, &shifts, &masks);
        assert_eq!(0b10001000101010010, state_index);
    }
}