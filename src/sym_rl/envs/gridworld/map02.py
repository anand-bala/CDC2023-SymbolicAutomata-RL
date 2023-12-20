"""
Environment 2 from
Q-Learning for Robust Satisfaction of Signal Temporal Logic Specifications
https://arxiv.org/pdf/1609.07409.pdf

4x4 grid
 start at (2,2)
Need to oscillate between A=(2,2) and B=(1,3) every 3 timesteps
F[0,2]A  and  F[0,2]B

"""

from typing import Optional

from rtamt_helpers.to_automaton import get_stl_formula, get_symbolic_automaton

from sym_rl.envs.gridworld.gridworld import GridWorldEnv
from sym_rl.utils import make_product_system
from sym_rl.wrappers.simple_product import ProductSystem

SHAPE = (4, 4)
INITIAL = (1, 2)

ROI = {
    "A": [(2, 2)],
    "B": [(1, 3)],
}

A = "(( x == 2 ) and ( y == 2 ))"
B = "(( x == 1 ) and ( y == 3 ))"

SPECIFICATIONS = {
    "bounded_recurrence1": f"G[0,10] ((F[0,2] {A}) and (F[0,2] {B}))",
    "bounded_recurrence2": f"G[0,10] ((F[0,5] {A}) and (F[0,5] {B}))",
    "unbounded_recurrence": f"G ((F {A}) and (F {B}))",
}


def _make_map02_creator(stl_spec: str):
    def make_fn(
        *,
        reward_method: str = "sparse",
        prob_of_slip: float = 0.1,
        d_max: float = 8,
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


GridWorldMap02_BoundedRecurrence1 = _make_map02_creator(
    f"G[0,10] ((F[0,2] {A}) and (F[0,2] {B}))"
)


GridWorldMap02_BoundedRecurrence2 = _make_map02_creator(
    f"G[0,10] ((F[0,5] {A}) and (F[0,5] {B}))"
)


GridWorldMap02_UnboundedRecurrence = _make_map02_creator(f"G ((F {A}) and (F {B}))")
