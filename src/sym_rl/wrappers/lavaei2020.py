from typing import Dict

import numpy as np

from sym_rl.wrappers.simple_product import ProductSystem
from sym_rl.wrappers.symbolic_product.eta import distance_to_accepting_set


class Lavaei2020RewardShaping(ProductSystem):
    def _post_init(self, **kwargs):
        eta_calc = distance_to_accepting_set(self.automaton)
        self.eta_max = 1 + max([eta for eta in eta_calc.values() if np.isfinite(eta)])
        self.eta = {q: min(eta, self.eta_max) for q, eta in eta_calc.items()}

        self.kappa = kwargs.get("kappa", 1.0)

    def _nonvec_compute_reward(self, achieved_goal, desired_goal, info: Dict) -> float:
        assert "last_automaton_location" in info.keys()

        base_reward = super()._nonvec_compute_reward(achieved_goal, desired_goal, info)

        pre_pot = self.task_potential(info["last_automaton_location"])
        post_pot = self.task_potential(achieved_goal)

        return base_reward + self.reward_scaling * (post_pot - pre_pot)

    def task_potential(self, q) -> float:
        if self.eta[q] == 0:
            return 1
        else:
            d = self.eta[q]
            d0 = self.eta[self.automaton.initial]
            return self.kappa * (d - d0) / (1 - self.eta_max)
