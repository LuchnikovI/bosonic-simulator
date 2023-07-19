use rayon::prelude::{
    IntoParallelIterator,
    IndexedParallelIterator, ParallelIterator,
};
use num_complex::ComplexFloat;
use crate::subroutines_utils::{
    get_diagonal,
    get_size,
    get_operator_index,
    masks_and_offsets,
    Term, get_global_offset,
};

pub(super) unsafe fn apply_term<const N: usize, T: ComplexFloat + Send + Sync>(
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
        *dst = *dst + delta * *diagonal.get_unchecked(operator_index) * *src;
    });
}