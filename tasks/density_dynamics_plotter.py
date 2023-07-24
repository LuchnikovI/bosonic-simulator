#!/usr/bin/python3

import os
from math import sqrt
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser(
        prog='Debug plotter.',
    )
    parser.add_argument(
        "--path", "-p",
        help="Path to the result  *.pickle file",
        default=os.getcwd(),
    )
    args = parser.parse_args()
    dirname = os.path.dirname(args.path)
    with open(args.path, 'rb') as f:
        result = pickle.load(f)
    densities = []
    for boson in result:
        boson = np.array(boson).astype(np.complex128)
        boson = boson[..., 0] + 1j * boson[..., 1]
        dim = int(sqrt(boson.shape[-1]))
        boson = boson.reshape((boson.shape[0], dim, dim))
        boson = boson * np.arange(dim).reshape((1, 1, -1))
        boson = np.trace(boson, axis1=1, axis2=2).real
        densities.append(boson)
    densities = np.array(densities)
    plt.figure()
    plt.plot(densities.T)
    plt.savefig(f"{dirname}/density_dynamics_v1.pdf")
    plt.figure()
    plt.imshow(densities)
    plt.savefig(f"{dirname}/density_dynamics_v2.pdf")

if __name__ == "__main__":
    main()