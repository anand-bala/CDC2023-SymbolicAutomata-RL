#!/usr/bin/env python

import argparse
import logging
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_utils import create_data_table, load_data, plot_curve, set_size
from experiments.map_registry import get_map_spec
from mdp.gridworld import GridWorld
from mdp.mdp import MDP
from product_mdp.rewards.reward_registry import make_reward
from product_mdp.rewards.true_potential import TruePotentialRewards
from rtamt_helpers.to_automaton import get_stl_formula, get_symbolic_automaton
from syma.automaton import SymbolicAutomaton

MAX_AUT_STATES = 9
MAX_PLOT_COLS = 3


logging.basicConfig(
    level=logging.INFO,
    format="{levelname:8s} {name:15s} {message}",
    style="{",
)
LOG = logging.getLogger("SymPot Plot")

plt.style.use(["science", "ieee"])
plt.rcParams.update(
    {
        "font.family": "serif",  # specify font family here
        "font.serif": ["Times"],  # specify font here
        "savefig.format": "pdf",
    }
)


@dataclass
class Args:
    map: str
    spec: str
    outfile: Optional[Path]


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Plot the symbolic potential for a given product MDP"
    )
    parser.add_argument("map", type=str, choices=["map01", "map02", "map03", "map04"])
    parser.add_argument(
        "spec",
        type=str,
        choices=[
            "bounded_reach",
            "unbounded_reach",
            "bounded_recurrence1",
            "bounded_recurrence2",
            "sequential",
            "branch_reach",
        ],
    )
    parser.add_argument(
        "-o",
        "--output",
        type=lambda p: Path(p).resolve(),
        help="Output file for PDF plot",
    )

    args = parser.parse_args()

    return Args(map=args.map, spec=args.spec, outfile=args.output)


def _get_subplots_spec(num_plots: int) -> Tuple[int, int]:
    nrows = (num_plots // MAX_PLOT_COLS) + 1
    ncols = (num_plots % MAX_PLOT_COLS) + 1

    return nrows, ncols


def plot_symbolic_potential(
    world: GridWorld,
    aut: SymbolicAutomaton,
):
    nrows, ncols = _get_subplots_spec(len(aut))
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)

    potential_fn = TruePotentialRewards(
        aut,
        world.unpack,
        d_max=world.worse_case_diameter(),
        episode_length=100,  # placeholder
    )

    mask = world._obs_map
    heatmap = dict()

    for q in aut.locations:
        if aut.is_accepting(q):
            continue
        LOG.info(f"Plotting for location {q}")
        heatmap[q] = np.zeros_like(mask, dtype=float)
        for i, j in world.states:
            heatmap[q][i, j] = potential_fn.task_potential(((i, j), q))

    for q, ax in zip(heatmap.keys(), axs.flat):
        sns.heatmap(
            heatmap[q],
            ax=ax,
            square=True,
            # cbar=False,
            # annot=True,
            # fmt="f",
            xticklabels=False,
            yticklabels=False,
        )
    plt.show()


def main():
    args = parse_args()

    mapname = args.map
    spec = args.spec
    outfile = args.outfile
    if outfile is None:
        outfile = Path.cwd() / f"{mapname}-{spec}.pdf"

    rng = np.random.default_rng(0)

    LOG.info(f"Loading map: {mapname}")
    map_spec = get_map_spec(mapname)
    world = map_spec.make_map(rng)

    LOG.info(f"Attempting to load specification: {spec}")
    specification = map_spec.specifications[spec]
    stl_spec = get_stl_formula(specification.formula, ("x", "int"), ("y", "int"))
    spec_aut = get_symbolic_automaton(stl_spec)

    if len(spec_aut) > MAX_AUT_STATES:
        LOG.critical(
            f"Number of states in automaton ({len(spec_aut)}) greater than {MAX_AUT_STATES}"
        )
        sys.exit(1)

    plot_symbolic_potential(world, spec_aut)


if __name__ == "__main__":
    main()
