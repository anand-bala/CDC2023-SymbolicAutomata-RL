from rtamt_helpers.to_automaton import SymbolicAutomaton
from stable_baselines3.common.type_aliases import GymEnv

from sym_rl.wrappers import (
    Lavaei2020RewardShaping,
    ProductSystem,
    SymbolicPotentialRewardShaping,
)
from sym_rl.wrappers.base import ValuationFn


def make_product_system(
    method: str,
    env: GymEnv,
    automaton: SymbolicAutomaton,
    valuation_fn: ValuationFn,
    *,
    d_max: float,
    penalize_rejecting: bool = False,
    reward_scaling: float = 1.0,
    **kwargs,
) -> ProductSystem:
    if method == "sparse":
        reward_class = ProductSystem
    elif method == "symbolic":
        reward_class = SymbolicPotentialRewardShaping
    elif method == "lavaei2020":
        reward_class = Lavaei2020RewardShaping
    else:
        raise ValueError(
            f"Unknown reward method: {method}. Must be one of (sparse, symbolic, lavaei2020)"
        )

    return reward_class(
        env,
        automaton,
        valuation_fn,
        d_max=d_max,
        penalize_rejecting=penalize_rejecting,
        reward_scaling=reward_scaling,
        **kwargs,
    )
