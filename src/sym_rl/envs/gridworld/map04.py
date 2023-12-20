"""A 16x16 grid

Here, the grid will have a few obstacles:

1. A large one in the center
2. An L shaped wall blocking a goal.

The goal here will be something like:

    F( (A | B) & F (C))
"""

from typing import Optional

from rtamt_helpers.to_automaton import get_stl_formula, get_symbolic_automaton

from sym_rl.envs.gridworld.gridworld import GridWorldEnv
from sym_rl.utils import make_product_system
from sym_rl.wrappers.simple_product import ProductSystem

SHAPE = (16, 16)
INITIAL = (0, 0)

UPPER_WALL = [(13, i) for i in range(2, 16)]  # type: list[tuple[int, int]]
MIDDLE_BLOCK = [(i, j) for i in range(4, 11) for j in range(4, 11)]

ROI = dict(
    A=[(row, col) for row in [14, 15] for col in [0, 1]],
    B=[(row, col) for row in [14, 15] for col in [5, 6, 7, 8]],
    C=[(row, col) for row in [0, 1] for col in [14, 15]],
    D=[(row, col) for row in [14, 15] for col in [14, 15]],
)

A = "((x <= 1) & (y >= 14))"
B = "((x >= 5) & (x <= 8) & (y >= 14))"
C = "((x >= 14) & (y <= 1))"
GOAL = "((x >= 14) & (y >= 14))"


SPECIFICATIONS = {
    "branch_reach": f"(F ({A} & F ({B} & F {GOAL}))) | (F ({C} & F {GOAL}))",
}


def _make_map04_creator(stl_spec: str):
    def make_fn(
        *,
        reward_method: str = "sparse",
        prob_of_slip: float = 0.1,
        d_max: float = 50,
        penalize_rejecting: bool = False,
        reward_scaling: float = 1.0,
        render_mode: Optional[str] = None,
        **kwargs,
    ) -> ProductSystem:
        env = GridWorldEnv(
            SHAPE,
            initial_pos=INITIAL,
            prob_of_slip=prob_of_slip,
            roi=ROI,
            walls=(UPPER_WALL + MIDDLE_BLOCK),
            render_mode=render_mode,
        )
        spec = get_stl_formula(stl_spec, ("x", "int"), ("y", "int"))
        aut = get_symbolic_automaton(spec)
        return make_product_system(
            reward_method,
            env,
            aut,
            GridWorldEnv.unpack,
            d_max=d_max,
            penalize_rejecting=penalize_rejecting,
            reward_scaling=reward_scaling,
            **kwargs,
        )

    return make_fn


GridWorldMap04_BranchReach = _make_map04_creator(
    f"(F ({A} & F ({B} & F {GOAL}))) | (F ({C} & F {GOAL}))"
)
