#!/usr/bin/python3

import os
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser(
        prog='Debug plotter.',
    )
    parser.add_argument(
        "--path", "-p",
        help="Path to the result  *.yaml file",
        default=os.getcwd(),
    )
    args = parser.parse_args()
    dirname = os.path.dirname(args.path)
    with open(args.path) as f:
        result = yaml.safe_load(f.read())
    dens = np.array(result[0]).astype(np.complex128)
    dens = dens[..., 0] + 1j * dens[..., 1]
    dens = dens.reshape([-1, 4, 4])
    lmbd = np.linalg.eigvalsh(dens)
    plt.figure()
    plt.plot(lmbd[:, 0])
    plt.plot(lmbd[:, 1])
    plt.plot(lmbd[:, 2])
    plt.plot(lmbd[:, 3])
    plt.savefig(f"{dirname}/zero_spin_dynamics.pdf")
    dens = np.array(result[1:]).astype(np.complex128)
    dens = dens[..., 0] + 1j * dens[..., 1]
    dens = dens.reshape((*dens.shape[:2], 2, 2))
    plt.figure()
    plt.imshow(dens[:, :, 0, 0].real)
    plt.savefig(f"{dirname}/rest_spins_dynamics.pdf")

if __name__ == "__main__":
    main()