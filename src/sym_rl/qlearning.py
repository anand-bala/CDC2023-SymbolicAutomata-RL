import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import check_for_nested_spaces
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import (
    configure_logger,
    get_linear_fn,
    get_schedule_fn,
    safe_mean,
)


class _NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if np.issubdtype(obj, np.integer):
            return int(obj)
        if np.issubdtype(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def default_q_entry() -> List[float]:
    # HACK: Pickle is annoying
    n_actions = 9
    return np.ones(n_actions).tolist()


def _make_flat_array(
    observation: Union[int, float, np.ndarray, Dict[str, np.ndarray]]
) -> np.ndarray:
    """Make an observation into a hashable for entry into a dict"""
    if isinstance(observation, np.ndarray):
        return observation
    if isinstance(observation, Mapping):
        # Assume flat dict
        values = [_make_flat_array(v) for v in observation.values()]
        return np.concatenate(values).flatten()
    return np.atleast_1d(observation)


class TabularQLearning:
    """Tabular Q-Learning algorithm"""

    def __init__(
        self,
        env: Optional[Union[str, gym.Env]],
        learning_rate: Union[float, Schedule] = 0.5,
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        create_eval_env: bool = False,
        seed: Optional[int] = None,
    ):

        self.verbose = verbose
        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        self._episode_num = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.tensorboard_log = tensorboard_log

        self._eval_freq = 10
        self._n_eval_episodes = 10
        self._log_path: Optional[Union[str, Path]] = None

        if create_eval_env:
            eval_env: GymEnv = maybe_make_env(env, self.verbose)  # type: ignore
            assert eval_env is not None
            assert isinstance(eval_env, gym.Env)
            self.eval_env = eval_env

        env = maybe_make_env(env, self.verbose)  # type: ignore
        assert isinstance(env, gym.Env)

        # if env.num_envs > 1:
        #     raise ValueError(
        #         "Error: the model does not support multiple envs; it requires a single vectorized environment."
        #     )
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Check supported actions
        assert isinstance(
            self.action_space, spaces.Discrete
        ), f"Unsupported action space {self.action_space}"
        self._n_actions = self.action_space.n
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # For updating the target network with multiple envs:
        self._n_calls = 0

        self._logger = None

        # Finally, our q-table
        self.q_table: Dict[Tuple, List[float]] = defaultdict(default_q_entry)

        self._setup_model()

    def _setup_model(self):
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        # Setup information buffers
        self.ep_success_buffer = deque(maxlen=100)
        self.ep_info_buffer = deque(maxlen=100)
        self._current_progress_remaining = 1.0

    @property
    def current_progress_remaining(self) -> float:
        self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(
            self._total_timesteps
        )
        return self._current_progress_remaining

    @property
    def np_random(self) -> np.random.Generator:
        return self.env.np_random

    def predict(self, observation, deterministic: bool = False) -> int:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        state = tuple(_make_flat_array(observation))

        if not deterministic and self.np_random.uniform() < self.exploration_rate:
            return self.np_random.integers(self._n_actions)
        else:
            return int(np.argmax(self.q_table[state]))

    @staticmethod
    def _wrap_env(env: gym.Env, monitor_wrapper: bool = True) -> gym.Env:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.
        :param env:
        :param verbose:
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not is_wrapped(env, Monitor) and monitor_wrapper:  # type: ignore
            env = Monitor(env)
        # Make sure that dict-spaces are not nested (not supported)
        check_for_nested_spaces(env.observation_space)

        return env

    def set_logger(self, logger: logger.Logger) -> None:
        """
        Setter for for logger object.
        .. warning::
          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Optional[logger.Logger]:
        """Getter for the logger object."""
        return self._logger

    def _serialize_q_keys(self):
        return [{"state": k, "q": v} for k, v in self.q_table.items()]

    def _deserialize_q_keys(self, encoded: List[Dict]):
        def to_entry_key(e):
            if isinstance(e, list):
                return tuple(to_entry_key(i) if isinstance(i, list) else i for i in e)
            else:
                return e

        for entry in encoded:
            state = to_entry_key(entry["state"])
            if not np.isscalar(state):
                state = tuple(state)
            qvalues = entry["q"]
            assert isinstance(qvalues, list)
            assert len(qvalues) == self._n_actions
            self.q_table[state] = qvalues  # type: ignore

    def save(self, save_file: Path):
        new_q_table = self._serialize_q_keys()
        new_dict = {
            "qtable": new_q_table,
            "seed": self.seed,
            "rng_state": self.env.np_random.bit_generator.state,
            "total_timesteps": self._total_timesteps,
            "episode_num": self._episode_num,
            "num_timesteps_at_start": self._num_timesteps_at_start,
            "gamma": self.gamma,
            "epsilon": self.exploration_rate,
            "epsilon_end": self.exploration_final_eps,
            "epsilon_decay": self.exploration_fraction,
            "n_calls": self._n_calls,
        }
        with open(save_file, "w") as model:
            json.dump(new_dict, model, cls=_NumpyEncoder)

    def load(self, file: Path):
        with open(file, "r") as model:
            chk = json.load(model)
        self._deserialize_q_keys(chk["qtable"])
        self.seed = chk["seed"]
        self.np_random.bit_generator.state = chk["rng_state"]
        self._total_timesteps = chk["total_timesteps"]
        self._episode_num = chk["episode_num"]
        self._num_timesteps_at_start = chk["num_timesteps_at_start"]
        self.gamma = chk["gamma"]
        self.exploration_initial_eps = chk["epsilon"]
        self.exploration_final_eps = chk["epsilon_end"]
        self.exploration_fraction = chk["epsilon_decay"]
        self._n_calls = chk["n_calls"]

        self._setup_model()

    def _setup_logging(
        self,
        log_path: Optional[str] = None,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
    ):
        if not self._custom_logger:
            self._logger = configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
            )
        if log_path is not None:
            self._log_path = Path(log_path)
            seed_str = f"{self.seed:05d}" if self.seed is not None else "None"
            self._chk_dir = self._log_path / f"{tb_log_name}-seed{seed_str}-chk"

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env],
        eval_freq: int = 10,
        n_eval_episodes: int = 10,
        reset_num_timesteps: bool = True,
    ):
        self.start_time = time.time_ns()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        if eval_env is not None:
            assert isinstance(eval_env, gym.Env)
            self.eval_env = self._wrap_env(eval_env)

        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes

        # Create checkpoint directory if needed
        if not self._chk_dir.is_dir():
            self._chk_dir.mkdir(parents=True, exist_ok=True)

    def _get_resume_point(self, reset_num_timesteps: bool) -> Optional[int]:
        """Get the last iteration run in this experiment"""
        import re

        if reset_num_timesteps:
            return None

        checkpoints_dir = self._chk_dir
        if not checkpoints_dir.is_dir():
            # If the checkpoints_dir doesn't exist, don't load anything and return as is
            return None

        checkpoint_file_re = re.compile(r"model_(\d{7})")

        # Iterate over the checkpoint files and find the last one
        chk_iter = -1
        for item in checkpoints_dir.iterdir():
            if not item.is_file():
                continue
            match = checkpoint_file_re.match(item.stem)
            if match is None:
                continue
            chk_iter = max(chk_iter, int(match.group(1)))

        if chk_iter == -1:
            return None
        return chk_iter

    def update(self, old_state, action, new_state, reward, done: bool):
        """Update the Q-learner with the observation, reward, and if the episode ended."""
        assert self.logger is not None

        s = tuple(_make_flat_array(old_state).tolist())
        a = action
        s_ = tuple(_make_flat_array(new_state).tolist())

        self.logger.debug(f"(s, a, r, s') = ({s}, {a}, {reward}, {s_})")

        alpha = self.lr_schedule(self.current_progress_remaining)
        self.logger.record("train/learning_rate", alpha)

        td = reward + self.gamma * (np.amax(self.q_table[s_]) - self.q_table[s][a])
        self.q_table[s][a] += alpha * td

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10,
        n_eval_episodes: int = 10,
        tb_log_name: str = "TabularQ",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        resume: bool = False,
    ):
        if resume:
            reset_num_timesteps = False

        self._setup_logging(
            log_path=eval_log_path,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
        )

        initial_iter = 0
        if resume:
            last_chk = self._get_resume_point(reset_num_timesteps)
            if last_chk is not None:
                initial_iter = last_chk + 1
                chk_file = self._chk_dir / f"model_{last_chk:07}.json"
                self.load(chk_file)
        self.num_timesteps = initial_iter
        assert self.logger is not None
        self.logger.info(f"Start training from iteration {self.num_timesteps}")

        self._setup_learn(
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            reset_num_timesteps=reset_num_timesteps,
        )

        while self.num_timesteps < self._total_timesteps:
            self._run_episode()

            if self._episode_num % log_interval == 0:
                self._dump_logs()
            if (
                self._episode_num % self._eval_freq == 1
            ):  # Start eval from the first episode
                self._evaluate_policy()

    def _run_episode(self):
        assert self.logger is not None

        self.logger.info(f"Running episode {self._episode_num+1}")
        obs = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.predict(obs)
            new_obs, reward, done, info = self.env.step(action)  # type: ignore
            self.num_timesteps += 1
            self._n_calls += 1
            total_reward += reward

            self._update_info_buffers(done, info)
            self.update(obs, action, new_obs, reward, done)

            self.exploration_rate = self.exploration_schedule(
                self.current_progress_remaining
            )
            self.logger.record("rollout/exploration_rate", self.exploration_rate)

            obs = new_obs
        self.logger.info(f"Completed episode {self._episode_num + 1}")
        self.logger.info(f"Total Reward = {total_reward}")
        self._episode_num += 1
        return total_reward

    def _evaluate_policy(self):
        """Runs a bunch of episodes and computes the average total reward, along with approximate probability of satisfaction."""
        assert self.logger is not None
        self.logger.info(f"Evaluating policy at iteration {self.num_timesteps}")

        reward_list: Deque[float] = deque()
        success_list: Deque[bool] = deque()

        for _ in range(self._n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.predict(obs, deterministic=True)
                step_returns = self.eval_env.step(action)
                if len(step_returns) == 4:
                    obs, reward, done, info = step_returns
                else:
                    assert len(step_returns) == 5
                    obs, reward, terminated, truncated, info = step_returns
                    done = terminated or truncated
                total_reward += reward

                if done:
                    success: Optional[bool] = info.get("is_success")
                    if success is not None:
                        success_list.append(success)
            reward_list.append(total_reward)

        eval_avg_reward = sum(reward_list) / len(reward_list)
        self.logger.record("eval/mean_reward", eval_avg_reward)
        self.logger.info(f"Average Total Reward: {eval_avg_reward}")
        if len(success_list) > 0:
            eval_success_rate = sum(success_list) / len(success_list)
            self.logger.record("eval/success_rate", eval_success_rate)
            self.logger.info(f"Success Rate: {eval_success_rate}")

        checkpoint_file = self._chk_dir / f"model_{self.num_timesteps:07}.json"
        self.save(checkpoint_file)

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        assert self.logger is not None

        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )

        # if len(self.ep_success_buffer) > 0:
        #     self.logger.record(
        #         "rollout/success_rate", safe_mean(self.ep_success_buffer)
        #     )
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _update_info_buffers(self, done, info):
        maybe_ep_info = info.get("episode")
        maybe_is_success = info.get("is_success")
        if maybe_ep_info is not None:
            self.ep_info_buffer.extend([maybe_ep_info])
        if maybe_is_success is not None and done:
            self.ep_success_buffer.append(maybe_is_success)
