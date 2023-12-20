from typing import Dict

import numpy as np

from sym_rl.wrappers.simple_product import ProductSystem

from .eta import distance_to_accepting_set
from .symbolic_dist import symbolic_distance
from .vpd import value_predicate_distance as vpd


class SymbolicPotentialRewardShaping(ProductSystem):
    """Computes the symbolic distance-based potential function for the given symbolic
    automaton"""

    def _post_init(self, **kwargs):
        eta_calc = distance_to_accepting_set(self.automaton)
        self.eta_max = 1 + max([eta for eta in eta_calc.values() if np.isfinite(eta)])
        self.eta = {q: min(eta, self.eta_max) for q, eta in eta_calc.items()}

        self._symbolic_goal_dist = symbolic_distance(
            self.automaton, self.automaton.accepting_sink()
        )
        # Remove any inf in the goal dist
        for e, dist in self._symbolic_goal_dist.items():
            if np.isinf(dist):
                self._symbolic_goal_dist[e] = abs(self.max_reward)

    def _nonvec_compute_reward(self, achieved_goal, desired_goal, info: Dict) -> float:
        assert "last_observation" in info.keys()
        assert "current_observation" in info.keys()
        assert "last_automaton_location" in info.keys()

        base_reward = super()._nonvec_compute_reward(achieved_goal, desired_goal, info)

        pre_pot = self.task_potential(
            info["last_automaton_location"], info["last_observation"], info
        )
        post_pot = self.task_potential(achieved_goal, info["current_observation"], info)

        return base_reward + self.reward_scaling * (pre_pot - post_pot)

    def task_potential(self, location: int, observation, info: Dict) -> float:
        r"""
        The task potential for a state `(s,q)` is defined as:

        .. math::
            \Phi(s, q) = \min_{(q, q') \in E, q \not= q'} vpd(s, \psi(q, q')) + dist_{sym} (q, q')
        """
        s, q = observation, location
        val = self._val_fn(s, info)

        dist = np.inf
        # For each outgoing transition from q (that is not a self-loop), we need to
        # compute the vpd of s to the guard on that transiton. Then, we add this vpd
        # value to the symbolic distance of that transition from the goal state.
        #
        # UPDATE: I realized that it is not just regular self-loops, but also
        # transitions that are "stationary" in general, i.e., where eta(q) == eta(q_)
        for q_ in self.automaton._graph.successors(q):
            guard = self.automaton.get_guard(q, q_)
            cur_vpd = vpd(val, guard)
            if cur_vpd == 0 and self.eta[q] == self.eta[q_]:
                # Since the automaton is complete, there is at least 1 transition
                # where `cur_vpd` == 0. This can happen either in a self-loop on a
                # location in the automaton, or (as in the case of a bounded-time
                # specification) due to a vacuous transition that happens due to time
                # passing.
                #
                # While the former never actually contributes to the symbolic distance
                # travelled, the latter doesn't do so when there is not actual
                # progress made towards the goal in the automaton. For this case, we
                # need to filter the VPDs based on the eta values.
                #
                # FIXME: There may be a corner case class of symbolic automata where
                # both the above conditions are true, but we do want to take the
                # distance into account. But I think there is an argument that can be
                # made that these don't matter.
                continue
            else:
                cur_dist = cur_vpd + self._symbolic_goal_dist[(q, q_)]  # type: ignore
                dist = min(dist, cur_dist)

        if np.isinf(dist):
            # This implies that we moved into a rejecting sink.
            # HACK: Just return worst case distance
            dist = self.max_reward
        return dist
