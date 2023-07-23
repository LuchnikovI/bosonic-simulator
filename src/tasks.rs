use std::fmt::Debug;
use num_traits::Zero;
use num_complex::ComplexFloat;
use serde::{Serialize, Deserialize};
use log::error;
use indicatif::ProgressIterator;

use crate::chebyshev::{cheb_exp, FromComplex64};
use crate::subroutines_utils::{TrueComplex, Value, Op, Term};
use crate::subroutines::{
    init_std,
    apply_term,
    add_inplace,
    get_density,
    init_zero,
    set2zero,
};

#[derive(
    Deserialize,
    Serialize,
    Debug,
    Clone,
    PartialEq,
    PartialOrd
)]
#[serde(untagged)]
enum DensEnum {
    One([usize; 1]),
    Two([usize; 2]),
    Three([usize; 3]),
    Four([usize; 4]),
}

#[derive(
    Deserialize,
    Serialize,
    Debug,
    Clone,
    PartialEq,
    PartialOrd
)]
#[serde(untagged)]
pub enum TermAndAmpl<T: ComplexFloat> {
    One {
        ampl: T::Real,
        pos: [usize; 1],
        ops: [Op; 1],
    },
    Two {
        ampl: T::Real,
        pos: [usize; 2],
        ops: [Op; 2],
    },
    Three {
        ampl: T::Real,
        pos: [usize; 3],
        ops: [Op; 3],
    },
    Four {
        ampl: T::Real,
        pos: [usize; 4],
        ops: [Op; 4],
    },
}

#[derive(
    Deserialize,
    Serialize,
    Debug,
    Clone,
    PartialEq,
    PartialOrd
)]
pub struct ChebyshevDynamics<T>
where
    T: ComplexFloat
{
    qubits_per_mode: Vec<usize>,
    total_time_steps_number: usize,
    time_step_size: T::Real,
    hamiltonian: Vec<TermAndAmpl<T>>,
    density_matrices: Vec<DensEnum>,
}

#[derive(
    Deserialize,
    Serialize,
    Debug,
    Clone,
    PartialEq,
    PartialOrd,
)]
pub enum Task<T>
where
    T: ComplexFloat,
    T::Real: Serialize + for<'a > Deserialize<'a> + Debug
{
    ChebyshevDynamics(ChebyshevDynamics<T>)
}

impl<T> ChebyshevDynamics<T>
where
    T: Value + TrueComplex + FromComplex64 + std::iter::Sum + Debug,
    T::Real: Value,
{
    pub fn run(&self, order: usize, acc: T::Real) -> Vec<Vec<Vec<T>>>
    {
        let mut state = init_std::<T>(&self.qubits_per_mode);
        let mut exp = init_zero::<T>(&self.qubits_per_mode);
        let mut aux = init_zero::<T>(&self.qubits_per_mode);
        let mut density_matrices = vec![
            Vec::with_capacity(self.total_time_steps_number + 1); self.density_matrices.len()
        ];
        for (dens, dst) in self.density_matrices.iter().zip(&mut density_matrices)
        {
            match dens
            {
                DensEnum::One(positions) => {
                    dst.push(get_density(&state, positions, &self.qubits_per_mode));
                },
                DensEnum::Two(positions) => {
                    dst.push(get_density(&state, positions, &self.qubits_per_mode));
                },
                DensEnum::Three(positions) => {
                    dst.push(get_density(&state, positions, &self.qubits_per_mode));
                },
                DensEnum::Four(positions) => {
                    dst.push(get_density(&state, positions, &self.qubits_per_mode));
                },
            }
        }
        for _ in (0..self.total_time_steps_number).progress() {
            let update_fn = |dst: &mut Vec<T>, src: &Vec<T>, delta| {
                for term in &self.hamiltonian {
                    match term {
                        TermAndAmpl::One { ampl, pos, ops } => {
                            apply_term(
                                dst, src,
                                &Term { positions: *pos, op_types: *ops },
                                &self.qubits_per_mode,
                                delta * <T as TrueComplex>::new(T::Real::zero(), *ampl * self.time_step_size),
                            );
                        },
                        TermAndAmpl::Two { ampl, pos, ops } => {
                            apply_term(
                                dst, src,
                                &Term { positions: *pos, op_types: *ops },
                                &self.qubits_per_mode,
                                delta * <T as TrueComplex>::new(T::Real::zero(), *ampl * self.time_step_size),
                            );
                        },
                        TermAndAmpl::Three { ampl, pos, ops } => {
                            apply_term(
                                dst, src,
                                &Term { positions: *pos, op_types: *ops },
                                &self.qubits_per_mode,
                                delta * <T as TrueComplex>::new(T::Real::zero(), *ampl * self.time_step_size),
                            );
                        },
                        TermAndAmpl::Four { ampl, pos, ops } => {
                            apply_term(
                                dst, src,
                                &Term { positions: *pos, op_types: *ops },
                                &self.qubits_per_mode,
                                delta * <T as TrueComplex>::new(T::Real::zero(), *ampl * self.time_step_size),
                            );
                        },
                    }
                }
            };
            cheb_exp::<Vec<T>, T>(
                &mut exp,
                &mut state, 
                &mut aux,
                update_fn,
                |dst, src, delta| add_inplace(dst.as_mut_slice(), src.as_slice(), delta),
                order
            );
            std::mem::swap(&mut exp, &mut state);
            set2zero(&mut exp);
            set2zero(&mut aux);
            for (dens, dst) in self.density_matrices.iter().zip(&mut density_matrices)
            {
                match dens
                {
                    DensEnum::One(positions) => {
                        let dens = get_density(&state, positions, &self.qubits_per_mode);
                        let dim = (dens.len() as f64).sqrt() as usize;
                        let trace = dens.iter().enumerate().filter(|(i, _)| i % (dim + 1) == 0).map(|(_, x)| *x).sum::<T>();
                        if (trace - T::one()).abs() > acc {
                            error!("Trace of a density matrix sufficiently deviates from 1, trace value: {:?}", trace);
                        }
                        dst.push(dens);
                    },
                    DensEnum::Two(positions) => {
                        let dens = get_density(&state, positions, &self.qubits_per_mode);
                        let dim = (dens.len() as f64).sqrt() as usize;
                        let trace = dens.iter().enumerate().filter(|(i, _)| i % (dim + 1) == 0).map(|(_, x)| *x).sum::<T>();
                        if (trace - T::one()).abs() > acc {
                            error!("Trace of a density matrix sufficiently deviates from 1, trace value: {:?}", trace);
                        }
                        dst.push(dens);
                    },
                    DensEnum::Three(positions) => {
                        let dens = get_density(&state, positions, &self.qubits_per_mode);
                        let dim = (dens.len() as f64).sqrt() as usize;
                        let trace = dens.iter().enumerate().filter(|(i, _)| i % (dim + 1) == 0).map(|(_, x)| *x).sum::<T>();
                        if (trace - T::one()).abs() > acc {
                            error!("Trace of a density matrix sufficiently deviates from 1, trace value: {:?}", trace);
                        }
                        dst.push(dens);
                    },
                    DensEnum::Four(positions) => {
                        let dens = get_density(&state, positions, &self.qubits_per_mode);
                        let dim = (dens.len() as f64).sqrt() as usize;
                        let trace = dens.iter().enumerate().filter(|(i, _)| i % (dim + 1) == 0).map(|(_, x)| *x).sum::<T>();
                        if (trace - T::one()).abs() > acc {
                            error!("Trace of a density matrix sufficiently deviates from 1, trace value: {:?}", trace);
                        }
                        dst.push(dens);
                    },
                }
            }
        }
        density_matrices
    }
}