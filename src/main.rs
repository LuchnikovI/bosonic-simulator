mod subroutines_utils;
mod subroutines;
mod tasks;
mod chebyshev;

#[cfg(test)]
mod test_utils;

#[cfg(test)]
mod subroutines_tests;

use std::fs::{read_to_string, write};
use chebyshev::FromComplex64;
use clap::Parser;
use num_complex::{
    Complex32,
    Complex64,
};
use subroutines_utils::TrueComplex;
use crate::subroutines_utils::Value;
use crate::tasks::Task;

/// This program simulates a bosonic system exactly. It takes
/// a task config in *.yaml format and produces results
/// also in *.yaml format.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {

    /// A config file path
    #[arg(short, long)]
    config: String,

    /// A result file path
    #[arg(short, long)]
    result: String,

    /// Precision of computation
    #[arg(short, long, default_value_t=String::from("f32"))]
    dtype: String,
}

fn run<T>(config: String, output_path: &str, order: usize, acc: T::Real)
where
    T: Value + TrueComplex + FromComplex64 + std::iter::Sum,
    T::Real: Value,
{
    let task = serde_yaml::from_str::<Task<T>>(&config).expect("Unable to recognize a config");
    match task {
        Task::ChebyshevDynamics(task) => {
            let density_matrices = task.run(order, acc);
            let yaml_string = serde_yaml::to_string(&density_matrices).unwrap();
            write(output_path, yaml_string).expect("impossible write results to a file");
        },
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let config = read_to_string(&args.config).expect(&format!("Could not read a config file {:?}", &args.config));
    match args.dtype.as_str() {
        "f32" => run::<Complex32>(config, &args.result, 7, 1e-3),
        "f64" => run::<Complex64>(config, &args.result, 14, 1e-8),
        other => { panic!("Data-type \"{}\" is not recognized", other) }
    }
}