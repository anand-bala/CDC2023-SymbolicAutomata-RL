"""This map is a 6x6 grid with one region of interest.

The initial state will be (0,0), and we will have one ROI (labelled 'A') as a 2x2 square
in the top right of the grid.

The primary specification to satisfy will be F[0,15](x >= 4 & y >= 4)
"""

from typing import Optional

from rtamt_helpers.to_automaton import get_stl_formula, get_symbolic_automaton

from sym_rl.envs.gridworld.gridworld import GridWorldEnv
from sym_rl.utils import make_product_system
from sym_rl.wrappers.simple_product import ProductSystem

SHAPE = (6, 6)
INITIAL = (0, 0)

ROI = {
    "A": [(row, col) for row in [4, 5] for col in [4, 5]],
}

SPECIFICATIONS = {
    "unbounded_reach": "F ((x >= 4) & (y >= 4))",
    "bounded_reach": "F[0,14] ((x >= 4) & (y >= 4))",
}


def _make_map01_creator(stl_spec: str):
    def make_fn(
        *,
        reward_method: str = "sparse",
        prob_of_slip: float = 0.1,
        d_max: float = 12,
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


GridWorldMap01_UnboundedReach = _make_map01_creator("F ((x >= 4) & (y >= 4))")
GridWorldMap01_UnboundedReach.metadata = GridWorldEnv.metadata

GridWorldMap01_BoundedReach = _make_map01_creator("F[0,14] ((x >= 4) & (y >= 4))")
GridWorldMap01_BoundedReach.metadata = GridWorldEnv.metadata
