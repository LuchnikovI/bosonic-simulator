use num_complex::{Complex, ComplexFloat};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum Op {
    Rising,
    Lowering,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Term {
    pos: usize,
    op: Op,
}


// ----------------------------------------------------------------------------------------

mod subroutines_utils {

    use super::Term;

    #[inline]
    pub(super) fn get_batch_size<const N: usize>(
        all_encodings: &[usize],
        terms: &[Term; N],
    ) -> usize
    {
        all_encodings.into_iter()
            .enumerate()
            .filter(|(encoding_pos, _)| {
                let mut predicate = true;
                for Term { pos, .. } in terms {
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
        terms: &[Term; N],
    ) -> [usize; N]
    {
        let mut strides = [0usize; N];
        for (stride, Term { pos, .. }) in strides.iter_mut().zip(terms.iter()) {
            *stride = all_encodings[..*pos].iter()
                .map(|x| 2usize.pow(*x as u32))
                .product();
        }
        strides
    }

    #[inline]
    pub(super) fn get_target_encodings<const N: usize>(
        all_encodings: &[usize],
        terms: &[Term; N],
    ) -> [usize; N]
    {
        let mut target_encodings = [0; N];
        for (Term { pos, .. }, encoding)
        in terms.into_iter().zip(target_encodings.iter_mut())
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

    #[cfg(test)]
    mod tests {
        use crate::subroutines::{Term, Op};
        use super::*;

        #[test]
        fn test_subroutines_utils() {
            let all_encodings = [1, 2, 3, 2, 1, 1, 2, 1, 2, 3];
            let terms = [];
            let size = get_batch_size(&all_encodings, &terms);
            assert_eq!(2usize.pow(18), size);
            let strides = get_strides(&all_encodings, &terms);
            let target_encodings = get_target_encodings(&all_encodings, &terms);
            let masks = get_masks(&strides);
            let batch_index = get_batch_index(usize::MAX, &masks, &target_encodings);
            assert_eq!(usize::MAX, batch_index);
            let batch_index = get_batch_index(0usize, &masks, &target_encodings);
            assert_eq!(0, batch_index);
            let batch_index = get_batch_index(512usize - 1, &masks, &target_encodings);
            assert_eq!(512usize - 1, batch_index);
            // --------------------------------------------------------------------------
            let all_encodings = [1, 2, 3, 2, 1, 1, 2, 1, 2, 3];
            let mut terms = [
                Term { pos: 5, op: Op::Rising },
                Term { pos: 2, op: Op::Lowering },
                Term { pos: 8, op: Op::Rising },
            ];
            terms.sort();
            let size = get_batch_size(&all_encodings, &terms);
            assert_eq!(2usize.pow(12), size);
            let strides = get_strides(&all_encodings, &terms);
            assert_eq!([8, 512, 8192], strides);
            let target_encodings = get_target_encodings(&all_encodings, &terms);
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
            let mut terms = [
                Term { pos: 7, op: Op::Rising },
                Term { pos: 0, op: Op::Rising },
                Term { pos: 3, op: Op::Lowering },
                Term { pos: 9, op: Op::Rising },
            ];
            terms.sort();
            let size = get_batch_size(&all_encodings, &terms);
            assert_eq!(2usize.pow(11), size);
            let strides = get_strides(&all_encodings, &terms);
            assert_eq!([1, 64, 4096, 32768], strides);
            let target_encodings = get_target_encodings(&all_encodings, &terms);
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

    }

}

// ----------------------------------------------------------------------------------------

use subroutines_utils::*;

unsafe fn apply_term<const N: usize, T: ComplexFloat>(
    dst: *mut Complex<T>,
    src: *const Complex<T>,
    all_encodings: &[usize],
    terms: &[Term; N],
    delta: Complex<T>,
)
{
    let mut terms = *terms;
    terms.sort();
    let size = get_batch_size(all_encodings, &terms);
    let target_encodings = get_target_encodings(all_encodings, &terms);
    let strides = get_strides(all_encodings, &terms);
    let masks = get_masks(&strides);
    for index in 0..size {
        let bindex = get_batch_index(index, &masks, &target_encodings);
        
    }
}
