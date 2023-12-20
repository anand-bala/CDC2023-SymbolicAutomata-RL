"""A 25x25 grid

Here we will create a 25x25 grid with initial state (10,10) and the following regions:

* A: (x >= 22) & (y >= 22)
* B: (x >= 22) & (y <= 2)
* C: (x <= 2) & (y >= 22)

The specification we want to satisfy is:

F(A & F(B & F(C)))
"""

from typing import Optional

from rtamt_helpers.to_automaton import get_stl_formula, get_symbolic_automaton

from sym_rl.envs.gridworld.gridworld import GridWorldEnv
from sym_rl.utils import make_product_system
from sym_rl.wrappers.simple_product import ProductSystem

SHAPE = (25, 25)
INITIAL = (10, 10)

ROI = dict(
    A=[(row, col) for row in [22, 23, 24] for col in [22, 23, 24]],
    B=[(row, col) for row in [0, 1, 2] for col in [22, 23, 24]],
    C=[(row, col) for row in [22, 23, 24] for col in [0, 1, 2]],
)


A = "((x >= 22) & (y >= 22))"
B = "((x >= 22) & (y <= 2))"
C = "((x <= 2) & (y >= 22))"

SPECIFICATIONS = {
    "sequential": f"F ({A} & F({B} & F({C})))",
    "bounded_sequential": f"F[0,50] ( {A} & F[0,50] ({B} & F[0,50] ({C})))",
}


def _make_map03_creator(stl_spec: str):
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


GridWorldMap03_Sequential = _make_map03_creator(f"F ({A} & F({B} & F({C})))")
GridWorldMap03_Sequential.metadata = GridWorldEnv.metadata

GridWorldMap03_BoundedSequential = _make_map03_creator(
    f"F[0,50] ( {A} & F[0,50] ({B} & F[0,50] ({C})))"
)
GridWorldMap03_BoundedSequential.metadata = GridWorldEnv.metadata
