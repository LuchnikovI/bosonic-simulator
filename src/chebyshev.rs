use num_complex::{ComplexFloat, Complex64, Complex32};
use num_traits::One;

/*use crate::tasks::TermAndAmpl;
use crate::subroutines::apply_term;*/

static BESSEL_COEFFS: [Complex64; 16] = [
    Complex64::new(1.2660658777520083355982446252147175376076703113549622068081353312, 0.),
    Complex64::new(0., -0.56515910399248502720769602760986330732889962162109200948029448),
    Complex64::new(-0.135747669767038281182852569994990922949871068112778187847546352, 0.),
    Complex64::new(0., 0.022168424924331902476285747629899615529415349169979258090109080),
    Complex64::new(0.0027371202210468663251380842155932297733789730929026393068918695, 0.),
    Complex64::new(0., -0.00027146315595697187518107390515377734238356442675814363497412),
    Complex64::new(-0.000022488661477147573327345164055456349543328825321202957150624, 0.),
    Complex64::new(0., 1.5992182312009952529319364883011478636185229037081491666241e-6),
    Complex64::new(9.9606240333639786298053219240279452669504669288868817881985e-8, 0.),
    Complex64::new(0., -5.51838586275867216308498045667662090644819508624808051273e-9),
    Complex64::new(-2.75294803983687362523571020100276353437157736403368652675e-10, 0.),
    Complex64::new(0., 1.248978308492491261356005467109383770504035818070745922593e-11),
    Complex64::new(5.1957611533928502524981733621192392626985642780454970518752e-13, 0.),
    Complex64::new(0., -1.9956316782072007564438602007663474563803913398266301430e-14),
    Complex64::new(-7.11879005412828574413684012673587610954679449625867991552e-16, 0.),
    Complex64::new(0., 2.370463051280748085544965280302145707288880874199766713661e-17),
];

pub trait FromComplex64 {
    fn new(val: Complex64) -> Self;
}

impl FromComplex64 for Complex64 {
    fn new(val: Complex64) -> Self {
        val
    }
}

impl FromComplex64 for Complex32 {
    fn new(val: Complex64) -> Self {
        Complex32::new(val.re as f32, val.im as f32)
    }
}

enum Sign {
    Pos,
    Neg,
}

struct Chebyshev<'a, T1>
{
    val: &'a mut T1,
    sign: Sign,
}

fn get_next_negative_chebyshev<T1, T2: ComplexFloat + FromComplex64>(
    prev_chebyshev: &mut Chebyshev<T1>,
    curr_chebyshev: &Chebyshev<T1>,
    update_fn: impl Fn(&mut T1, &T1, T2),
)
{
    match prev_chebyshev.sign {
        Sign::Pos => {
            match curr_chebyshev.sign {
                Sign::Pos => {
                    update_fn(&mut prev_chebyshev.val, &curr_chebyshev.val, -T2::new(Complex64::new(2., 0.)));
                    prev_chebyshev.sign = Sign::Neg
                },
                Sign::Neg => {
                    update_fn(&mut prev_chebyshev.val, &curr_chebyshev.val, T2::new(Complex64::new(2., 0.)));
                    prev_chebyshev.sign = Sign::Neg
                },
            }
        },
        Sign::Neg => {
            match curr_chebyshev.sign {
                Sign::Pos => {
                    update_fn(prev_chebyshev.val, curr_chebyshev.val, T2::new(Complex64::new(2., 0.)));
                    prev_chebyshev.sign = Sign::Pos
                },
                Sign::Neg => {
                    update_fn(prev_chebyshev.val, curr_chebyshev.val, -T2::new(Complex64::new(2., 0.)));
                    prev_chebyshev.sign = Sign::Pos
                },
            }
        },
    }
}

pub fn cheb_exp<T1, T2: ComplexFloat + FromComplex64>(
    exp: &mut T1,
    state: &mut T1,
    aux: &mut T1,
    update_fn: impl Fn(&mut T1, &T1, T2),
    add: impl Fn(&mut T1, &T1, T2),
    order: usize,
)
{
    let imag = Complex64::new(0., 1.);
    let mut imag_pow = imag;
    let two = Complex64::new(2., 0.);
    add(exp, state, T2::new(BESSEL_COEFFS[0]));
    update_fn(aux, state, T2::new(Complex64::one()));
    add(exp, aux, T2::new(two * imag_pow * BESSEL_COEFFS[1]));
    let mut prev_cheb = Chebyshev { val: state, sign: Sign::Pos };
    let mut curr_cheb = Chebyshev { val: aux, sign: Sign::Pos };
    for bessel in &BESSEL_COEFFS[2..order]
    {
        imag_pow = imag_pow * imag;
        get_next_negative_chebyshev(&mut prev_cheb, &curr_cheb, &update_fn);
        std::mem::swap(&mut curr_cheb, &mut prev_cheb);
        let coeff = match curr_cheb.sign {
            Sign::Pos => T2::new(two * imag_pow * *bessel),
            Sign::Neg => T2::new(-two * imag_pow * *bessel),
        };
        add(exp, curr_cheb.val, coeff);
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use num_complex::{Complex32, Complex64, ComplexFloat};
    use super::FromComplex64;

    use super::cheb_exp;

    fn _test_cheb_exp<T: ComplexFloat + FromComplex64 + Debug>(order: usize, acc: T::Real)
    {
        let x = T::new(Complex64::new(0.7, 0.2));
        let mut exp = T::zero();
        let mut aux = T::zero();
        let mut state = T::one();
        cheb_exp::<T, T>(&mut exp, &mut state, &mut aux,
            |dst, src, coeff| *dst = *dst + coeff * *src * x,
            |dst, src, coeff| *dst = *dst + coeff * *src,
            order,
        );
        assert!((exp - x.exp()).abs() < acc);
    }

    #[test]
    fn test_cheb_exp()
    {
        _test_cheb_exp::<Complex64>(14, 1e-10);
        _test_cheb_exp::<Complex32>(7, 1e-4);
    }
}