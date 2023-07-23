use rayon::prelude::{
    IntoParallelIterator,
    IndexedParallelIterator,
    ParallelIterator,
    IntoParallelRefMutIterator,
};
use rayon::ThreadPoolBuilder;
use num_cpus::get_physical;
use crate::subroutines_utils::{
    get_diagonal,
    get_size,
    get_operator_index,
    masks_and_offsets,
    get_global_offset,
    get_batch_size,
    get_strides,
    get_masks,
    get_target_encodings,
    get_density_size,
    get_batch_index,
    density_matrix_shifts_and_masks,
    density_matrix_index_to_state_index,
    Value,
    Term,
};

pub(super) fn apply_term<const N: usize, T: Value>(
    dst: &mut [T],
    src: &[T],
    term: &Term<N>,
    all_encodings: &[usize],
    delta: T,
)
{
    let size = 2usize.pow(get_size(all_encodings) as u32);
    let (masks, shifts) = masks_and_offsets(all_encodings, &term.positions);
    let (diagonal, _) = get_diagonal::<N, T>(term, all_encodings);
    let offset = get_global_offset(&term, &all_encodings);
    let (dst_iter, src_iter, enumerator) = if offset > 0 {
        let offset = offset as usize;
        (
            (&mut dst[..(size - offset)]).into_par_iter(),
            (&src[offset..]).into_par_iter(),
            (offset..size).into_par_iter(),
        )
    } else {
        let offset = -offset as usize;
        (
            (&mut dst[offset..]).into_par_iter(),
            (&src[..(size - offset)]).into_par_iter(),
            (0..(size - offset)).into_par_iter(),
        )
    };
    enumerator.zip(dst_iter.zip(src_iter)).for_each(|(index, (dst, src))| {
        let operator_index = get_operator_index(index, &shifts, &masks);
        *dst = *dst + delta * *unsafe { diagonal.get_unchecked(operator_index) } * *src;
    });
}

pub(super) fn get_density<T, const N: usize>(
    src: &[T],
    positions: &[usize; N],
    all_encodings: &[usize],
) -> Vec<T>
where
    T: Value,
{
    let mut positions = positions.clone();
    positions.sort();
    let batch_size = get_batch_size(&all_encodings, &positions);
    let strides = get_strides(&all_encodings, &positions);
    let target_encodings = get_target_encodings(&all_encodings, &positions);
    let masks = get_masks(&strides);
    let density_size = get_density_size(all_encodings, &positions);
    let (density_matrix_masks, shifts) = density_matrix_shifts_and_masks(&all_encodings, &positions);
    let threads_num = get_physical() + 1;
    let batch_size_per_thread = batch_size / threads_num + 1;
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(threads_num)
        .build()
        .unwrap();
    let mut density_per_thread = vec![vec![T::zero(); density_size * density_size]; threads_num];
    let mut stack_per_thread = vec![Vec::with_capacity(density_size); threads_num];
    thread_pool.scope(|s|{
        for (i, (density, stack)) in density_per_thread.iter_mut().zip(&mut stack_per_thread).enumerate() {
            s.spawn(move |_| {
                let start = i * batch_size_per_thread;
                let end = std::cmp::min((i + 1) * batch_size_per_thread, batch_size);
                for index in start..end
                {
                    let bi = get_batch_index(index, &masks, &target_encodings);
                    for j in 0..density_size {
                        let state_j = density_matrix_index_to_state_index(j, &shifts, &density_matrix_masks);
                        stack.push(unsafe { *src.get_unchecked(state_j + bi) });
                        for k in 0..=j {
                            unsafe {
                                *density.get_unchecked_mut(j * density_size + k) =
                                *density.get_unchecked_mut(j * density_size + k) +
                                (*stack.get_unchecked(j)).conj() *
                                *stack.get_unchecked(k)
                            }
                        }
                        for k in 0..j {
                            unsafe {
                                *density.get_unchecked_mut(k * density_size + j) =
                                *density.get_unchecked_mut(k * density_size + j) +
                                (*stack.get_unchecked(k)).conj() *
                                *stack.get_unchecked(j)
                            }
                        }
                    }
                    unsafe { stack.set_len(0); }
                }
            })
        }
    });
    density_per_thread.into_iter().reduce(|dens_acc, dens| {
        let mut dens_acc = dens_acc;
        for (dst, src) in dens_acc.iter_mut().zip(dens)
        {
            *dst = *dst + src;
        }
        dens_acc
    }).unwrap()
}

pub(super) fn init_std<T: Value>(
    all_encodings: &[usize],
) -> Vec<T>
{
    let size = 2usize.pow(get_size(&all_encodings) as u32);
    let mut state = Vec::with_capacity(size);
    unsafe { state.set_len(size) };
    state[1..].par_iter_mut().for_each(|x: &mut T| {
        *x = T::zero();
    });
    state[0] = T::one();
    state
}

pub(super) fn init_zero<T: Value>(
    all_encodings: &[usize],
) -> Vec<T>
{
    let size = 2usize.pow(get_size(&all_encodings) as u32);
    let mut state = Vec::with_capacity(size);
    unsafe { state.set_len(size) };
    state.par_iter_mut().for_each(|x: &mut T| {
        *x = T::zero();
    });
    state
}

pub(super) fn set2zero<T: Value>(
    state: &mut [T],
)
{
    state.par_iter_mut().for_each(|x: &mut T| {
        *x = T::zero();
    });
}

pub(super) fn add_inplace<T: Value>(
    dst: &mut [T],
    src: &[T],
    delta: T,
)
{
    dst.par_iter_mut().zip(src.into_par_iter()).for_each(|(d, s)| {
        *d = *d + *s * delta;
    });
}

pub(super) fn state_cpy<T: Value>(
    dst: &mut [T],
    src: &[T],
)
{
    dst.par_iter_mut().zip(src.into_par_iter()).for_each(|(d, s)| {
        *d = *s;
    });
}
