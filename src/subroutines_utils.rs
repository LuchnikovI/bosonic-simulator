use std::ops::Mul;
use num_complex::ComplexFloat;
use num_traits::One;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum Op {
    Rising,
    Lowering,
}

#[derive(Debug, Clone)]
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
            }
        }
        transposed_term
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
            Op::Rising => diag_pos += 1,
            Op::Lowering => diag_pos -= 1,
        };
    }
    diag_pos
}

#[inline]
fn diag_per_operator<T: ComplexFloat>(
    op: Op,
    size: usize,
) -> Vec<T>
{
    match op {
        Op::Rising => {
            (0..size).map(|x| T::from(x).unwrap().sqrt()).collect::<Vec<_>>()
        },
        Op::Lowering => {
            (1..size).chain(0..1).map(|x| T::from(x).unwrap().sqrt()).collect::<Vec<_>>()
        },
    }
}

#[inline]
pub(super) fn get_diagonal<const N: usize, T: ComplexFloat>(
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
            Op::Rising => diag_pos += 1,
            Op::Lowering => diag_pos -= 1,
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

#[cfg(test)]
mod tests {

    use num_complex::Complex64;
    use super::*;

    #[test]
    fn test_utils()
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
        let true_diag = (1..4).chain(0..1).map(|a| Complex64::from(a as f64).sqrt())
            .flat_map(|x| {
                (0..16).map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            })
            .flat_map(|x| {
                (1..8).chain(0..1).map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            });
        let all_encodings = [1, 3, 1, 4, 2, 2, 1];
        _test_get_diagonal(term, &all_encodings, true_diag);
        let term = Term {
            positions: [1, 2, 3, 7],
            op_types: [Op::Rising, Op::Lowering, Op::Rising, Op::Lowering]
        };
        let true_diag = (0..2).rev().map(|a| Complex64::from(a as f64).sqrt())
            .flat_map(|x| {
                (0..16).map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            })
            .flat_map(|x| {
                (0..2).rev().map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            })
            .flat_map(|x| {
                (0..8).map(|a| Complex64::from(a as f64).sqrt()).map(move |y| x * y)
            });
        let all_encodings = [1, 3, 1, 4, 2, 2, 1, 1, 2, 3, 4, 3, 1];
        _test_get_diagonal(term, &all_encodings, true_diag);
    }
}