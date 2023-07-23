use ndarray::{Array2, ArrayView};
use ndarray_einsum_beta::{einsum, ArrayLike};
use crate::subroutines_utils::{
    Term, Op, Value,
};

fn get_rising_op<T: Value>(dim: usize) -> Array2<T>
{
    let mut rising = Array2::zeros((dim, dim));
    for i in 1..dim {
        *rising.get_mut([i, i-1]).unwrap() = T::from(i).unwrap().sqrt();
    }
    rising
}

fn get_lowering_op<T: Value>(dim: usize) -> Array2<T>
{
    let mut lowering = Array2::zeros((dim, dim));
    for i in 1..dim {
        *lowering.get_mut([i-1, i]).unwrap() = T::from(i).unwrap().sqrt();
    }
    lowering
}

fn get_n_op<T: Value>(dim: usize) -> Array2<T>
{
    let mut lowering = Array2::zeros((dim, dim));
    for i in 0..dim {
        *lowering.get_mut([i, i]).unwrap() = T::from(i).unwrap();
    }
    lowering
}

fn get_nsq_op<T: Value>(dim: usize) -> Array2<T>
{
    let mut lowering = Array2::zeros((dim, dim));
    for i in 0..dim {
        *lowering.get_mut([i, i]).unwrap() = T::from(i).unwrap().powi(2);
    }
    lowering
}

pub(super) fn apply_term_test<const N: usize, T: Value>(
    dst: &mut [T],
    src: &[T],
    term: &Term<N>,
    all_encodings: &[usize],
    delta: T,
)
{
    // ------------------ C-layout to Fortran-layout --------------------------------------
    let all_encodings: Vec<_> = all_encodings.into_iter().rev().map(|x| *x).collect();
    let particles_number = all_encodings.len();
    let mut term = term.clone();
    term.op_types.reverse();
    term.positions.reverse();
    for pos in &mut term.positions {
        *pos = particles_number - *pos - 1;
    }
    // ------------------- Slices to Arrays -----------------------------------------------
    let mut shape = Vec::new();
    let mut operators = Vec::new();
    let mut start = 0;
    for (pos, op_type) in term.positions.iter().zip(term.op_types)
    {
        let dim = 2usize.pow((&all_encodings[start..*pos]).iter().map(|x| *x).sum::<usize>() as u32);
        shape.push(dim);
        let dim = 2usize.pow(all_encodings[*pos] as u32);
        shape.push(dim);
        start = pos + 1;
        match op_type {
            Op::Rising => {
                let op = get_rising_op::<T>(2usize.pow(all_encodings[*pos] as u32));
                operators.push(op);
            },
            Op::Lowering => {
                let op = get_lowering_op::<T>(2usize.pow(all_encodings[*pos] as u32));
                operators.push(op);
            },
            Op::N => {
                let op = get_n_op::<T>(2usize.pow(all_encodings[*pos] as u32));
                operators.push(op);
            },
            Op::N2 => {
                let op = get_nsq_op::<T>(2usize.pow(all_encodings[*pos] as u32));
                operators.push(op);
            },
        }
    }
    if start == all_encodings.len() {
        shape.push(1);
    } else {
        let dim = 2usize.pow((&all_encodings[start..]).iter().map(|x| *x).sum::<usize>() as u32);
        shape.push(dim);
    }
    let src = ArrayView::from_shape(shape.clone(), src).unwrap();
    // ----------------einsum-----------------------------------------------------------
    let mut operands: Vec<&dyn ArrayLike<T>> = Vec::new();
    operands.push(&src);
    for op in &operators {
        operands.push(op);
    }
    let mut free_state_indices_iter = (97u8..(97u8 + (2 * N as u8) + 1u8))
        .filter(|x| x % 2 != 0 )
        .map(|x| x as char);
    let mut free_ops_indices_iter = (97u8..(97u8 + (2 * N as u8) + 1u8))
        .filter(|x| x % 2 == 0 )
        .map(|x| x as char);
    let mut closed_indices_iter = ((122u8 - (N as u8) + 1u8)..=122u8).map(|x| x as char);
    let final_indices_iter = (97u8..(97u8 + (2 * N as u8) + 1u8))
        .map(|x| x as char);
    let mut einsum_string = String::new();
    let mut closed_indices_iter_clone = closed_indices_iter.clone();
    einsum_string.push(free_state_indices_iter.next().unwrap());
    for _ in 0..N {
        einsum_string.push(closed_indices_iter_clone.next().unwrap());
        einsum_string.push(free_state_indices_iter.next().unwrap());
    }
    einsum_string.push(',');
    for _ in 0..N {
        einsum_string.push(free_ops_indices_iter.next().unwrap());
        einsum_string.push(closed_indices_iter.next().unwrap());
        einsum_string.push(',');
    }
    einsum_string.pop();
    einsum_string.push_str("->");
    for c in final_indices_iter {
        einsum_string.push(c);
    };
    let result = einsum(&einsum_string.as_str(), &operands[..]).unwrap();
    for (d, s) in dst.iter_mut().zip(result.iter())
    {
        *d = *s * delta + *d;
    }
}

pub(super) fn get_density_test<const N: usize, T: Value>(
    src: &[T],
    positions: &[usize; N],
    all_encodings: &[usize],
) -> Vec<T>
{
    // ------------------ C-layout to Fortran-layout --------------------------------------
    let all_encodings: Vec<_> = all_encodings.into_iter().rev().map(|x| *x).collect();
    let particles_number = all_encodings.len();
    let mut positions = positions.clone();
    positions.reverse();
    for pos in &mut positions {
        *pos = particles_number - *pos - 1;
    }
    // ------------------- Slices to Arrays -----------------------------------------------
    let mut shape = Vec::new();
    let mut start = 0;
    for pos in positions.iter()
    {
        let dim = 2usize.pow((&all_encodings[start..*pos]).iter().map(|x| *x).sum::<usize>() as u32);
        shape.push(dim);
        let dim = 2usize.pow(all_encodings[*pos] as u32);
        shape.push(dim);
        start = pos + 1;
    }
    if start == all_encodings.len() {
        shape.push(1);
    } else {
        let dim = 2usize.pow((&all_encodings[start..]).iter().map(|x| *x).sum::<usize>() as u32);
        shape.push(dim);
    }
    let src = ArrayView::from_shape(shape.clone(), src).unwrap();
    let mut src_conj = src.to_owned();
    src_conj.iter_mut().for_each(|x| *x = x.conj());
    // ----------------einsum-----------------------------------------------------------
    let mut operands: Vec<&dyn ArrayLike<T>> = Vec::new();
    operands.push(&src_conj);
    operands.push(&src);
    let mut first_operand_free_indices_iter = (97u8..(97u8 + N as u8)).map(|x| x as char);
    let mut second_operand_free_indices_iter = ((97u8 + N as u8)..=(97u8 + 2 * N as u8)).map(|x| x as char);
    let mut closed_indices_iter = ((122u8 - (N as u8))..=122u8).map(|x| x as char);
    let mut einsum_string = String::new();
    let mut closed_indices_iter_clone = closed_indices_iter.clone();
    let mut first_operand_free_indices_iter_clone = first_operand_free_indices_iter.clone();
    let mut second_operand_free_indices_iter_clone = second_operand_free_indices_iter.clone();
    for _ in 0..N
    {
        einsum_string.push(closed_indices_iter_clone.next().unwrap());
        einsum_string.push(first_operand_free_indices_iter_clone.next().unwrap());
    }
    einsum_string.push(closed_indices_iter_clone.next().unwrap());
    einsum_string.push(',');
    for _ in 0..N
    {
        einsum_string.push(closed_indices_iter.next().unwrap());
        einsum_string.push(second_operand_free_indices_iter_clone.next().unwrap());
    }
    einsum_string.push(closed_indices_iter.next().unwrap());
    einsum_string.push_str("->");
    for _ in 0..N
    {
        einsum_string.push(first_operand_free_indices_iter.next().unwrap());
    }
    for _ in 0..N
    {
        einsum_string.push(second_operand_free_indices_iter.next().unwrap());
    }
    let result = einsum(&einsum_string.as_str(), &operands[..]).unwrap();
    result.iter().map(|x| *x).collect::<Vec<_>>()
}
