#!/usr/bin/env python

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from sym_rl.data_processing import load_data, plot_curve, set_size

plt.style.use(["science", "ieee"])
plt.rcParams.update(
    {
        "font.family": "serif",  # specify font family here
        "font.serif": ["Times"],  # specify font here
        "savefig.format": "pdf",
    }
)

METHODS = {
    "sparse": "Sparse",
    "lavaei2020": "Lavaei, et.al. 2020",
    "symbolic": "Our Method",
}


@dataclass
class Args:
    env: str
    logdir: Path
    outfile: Optional[Path]


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Create the prob of acceptance plot for the given map/spec"
    )
    parser.add_argument(
        "logdir",
        type=lambda p: Path(p).resolve(),
        help="Directory where logs are stored",
    )
    parser.add_argument(
        "env",
        type=str,
        choices=[
            "GridMap01BoundedReach-v0",
            "GridMap02BoundedRecurrence2-v0",
            "GridMap03Sequential-v0",
            "GridMap03BoundedSequential-v0",
            "GridMap04BranchReach-v0",
        ],
    )
    parser.add_argument(
        "-o",
        "--output",
        type=lambda p: Path(p).resolve(),
        help="Output file for PDF plot",
    )

    args = parser.parse_args()

    return Args(logdir=args.logdir, env=args.env, outfile=args.output)


def main():
    args = parse_args()

    logdir = args.logdir
    env = args.env
    outfile = args.outfile
    if outfile is None:
        outfile = Path.cwd() / f"{env}.pdf"

    assert logdir.is_dir()

    data = {method: load_data(logdir / env, method) for method in METHODS.keys()}

    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.set_ylabel("Probability of Satisfaction")
    ax1.set_xlabel("No. of Training Iterations")

    # Probability of accepting
    for method, label in METHODS.items():
        plot_curve(label, ax1, data[method], span=20, confidence=0.95, bernoulli=True)
    ax1.legend(loc="center right")

    # fig.set_size_inches(*set_size(347.12354, 0.20))
    fig.savefig(outfile)


if __name__ == "__main__":
    main()
