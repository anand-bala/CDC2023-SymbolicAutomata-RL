from collections import deque
from typing import Dict, TypeVar, Union

import gym
import numpy as np
from gym import spaces
from syma.automaton import SymbolicAutomaton

from sym_rl.wrappers.base import ValuationFn

State = TypeVar("State")
Action = TypeVar("Action")


class ProductSystem(gym.Wrapper):
    """Wraps an environment and supplements the observation space with the automaton
    location.

    Args:
        automaton: The symbolic automaton task specification.
        valuation_fn: A callable that maps a state to a dictionary of values.
    """

    def __init__(
        self,
        env: gym.Env,
        automaton: SymbolicAutomaton,
        valuation_fn: ValuationFn,
        *,
        d_max: float,
        penalize_rejecting: bool = False,
        reward_scaling: float = 1.0,
        **kwargs,
    ):
        super().__init__(env, **kwargs)

        assert automaton.is_complete(), "The Symbolic Automaton needs to be complete"
        assert (
            automaton.is_deterministic()
        ), "The Symbolic Automaton needs to be deterministic"
        assert (
            automaton.has_unique_accepting_sink()
        ), "The Symbolic Automaton needs to have a unique accepting sink"
        assert (
            len(set(automaton.rejecting())) <= 1
        ), "The Symbolic Automaton needs to have at most 1 rejecting sink"

        self._automaton = automaton
        self._val_fn = valuation_fn

        self.observation_space = spaces.Dict(
            {
                "observation": self.env.observation_space,
                "achieved_goal": spaces.Discrete(len(self._automaton)),
                "desired_goal": spaces.Discrete(len(self._automaton)),
            }
        )
        self.action_space = self.env.action_space

        self._last_observation = deque(maxlen=1)
        self._last_automaton_location = deque([self.automaton.initial] * 2, maxlen=2)

        self._penalize_rejecting = penalize_rejecting
        self._accepting_state = self.automaton.accepting_sink()

        # Sparse reward default
        self._min_reward = 0
        self._max_reward = 0

        self.max_reward = d_max
        if self._penalize_rejecting:
            self.min_reward = -d_max

        self.reward_scaling = reward_scaling

        self.metadata = self.env.metadata

        self._post_init(**kwargs)

    def _post_init(self, **kwargs):  # pyright: ignore
        return

    @property
    def min_reward(self) -> float:
        return self._min_reward

    @min_reward.setter
    def min_reward(self, value: float):
        self._min_reward = value
        self.reward_range = (self._min_reward, self._max_reward)

    @property
    def max_reward(self) -> float:
        return self._max_reward

    @max_reward.setter
    def max_reward(self, value: float):
        self._max_reward = value
        self.reward_range = (self._min_reward, self._max_reward)

    @property
    def automaton(self) -> SymbolicAutomaton:
        return self._automaton

    @property
    def current_automaton_location(self) -> int:
        assert len(self._last_automaton_location) == 2
        return self._last_automaton_location[-1]

    @property
    def last_automaton_location(self) -> int:
        assert len(self._last_automaton_location) == 2
        return self._last_automaton_location[0]

    def reset(self, **kwargs):
        self._last_automaton_location.clear()
        self._last_automaton_location.extend([self.automaton.initial] * 2)

        self._last_observation.clear()

        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs, info), info
        else:
            return self.observation(self.env.reset(**kwargs), {})

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        step_returns = self.env.step(action)
        observation, reward, *others, info = step_returns

        info["last_observation"] = self._last_observation[-1]
        info["current_observation"] = observation
        info["last_automaton_location"] = self.current_automaton_location

        observation = self.observation(observation, info)
        reward = self.compute_reward(
            observation["achieved_goal"], observation["desired_goal"], info
        )

        if self.automaton.is_accepting(observation["achieved_goal"]):
            assert observation["achieved_goal"] == observation["desired_goal"]
            info["is_success"] = True
        else:
            info["is_success"] = False

        return observation, reward, *others, info

    def observation(self, observation, info: Dict):
        # Get the next automaton location based on observation
        self._last_automaton_location.append(
            self._next_automaton_state(observation, info)
        )
        # Update observations
        self._last_observation.append(observation)

        return dict(
            observation=observation,
            achieved_goal=self.current_automaton_location,
            desired_goal=self._accepting_state,
        )

    def _nonvec_compute_reward(self, achieved_goal, desired_goal, info: Dict) -> float:
        assert "last_observation" in info.keys()
        assert "current_observation" in info.keys()
        assert "last_automaton_location" in info.keys()

        if self.automaton.is_accepting(achieved_goal):
            assert (
                achieved_goal == desired_goal
            ), "Desired goal is not the accepting state?"
            return self.max_reward
        if self._penalize_rejecting:
            if self.automaton.is_rejecting_sink(achieved_goal):
                return self.min_reward
        return 0

    def compute_reward(
        self,
        achieved_goal: Union[int, np.ndarray],
        desired_goal: Union[int, np.ndarray],
        info: Union[Dict, np.ndarray],
    ) -> Union[float, np.ndarray]:
        if not isinstance(info, Dict):
            assert isinstance(info, np.ndarray)
            assert isinstance(achieved_goal, np.ndarray)
            assert isinstance(desired_goal, np.ndarray)
            # Vectorized
            assert achieved_goal.shape[-1] == 1
            rewards = np.array(
                [
                    self._nonvec_compute_reward(ach, des, inf)
                    for ach, des, inf in zip(
                        achieved_goal.flatten(), desired_goal.flatten(), info
                    )
                ]
            )
            return rewards
        else:
            assert isinstance(achieved_goal, int)
            assert isinstance(desired_goal, int)
            return self._nonvec_compute_reward(achieved_goal, desired_goal, info)

    def _next_automaton_state(self, next_obs, info: Dict) -> int:
        out_edges = list(self.automaton.out_edges(self.current_automaton_location))

        q = self.current_automaton_location
        s_ = next_obs

        for _, q_ in out_edges:
            guard = self.automaton.get_guard(q, q_)
            values = self._val_fn(s_, info)
            ok = guard.check_sat(values)
            if ok:
                return q_
        raise RuntimeError(
            "BUG: No feasible next state for the deterministic, complete automaton."
        )
