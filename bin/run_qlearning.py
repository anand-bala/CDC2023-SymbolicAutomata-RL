import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import gym
from stable_baselines3.common.logger import Logger, make_output_format

import sym_rl.envs  # type: ignore
from sym_rl.qlearning import TabularQLearning


@dataclass
class Config:
    env: str
    method: str
    log_dir: Path
    seed: int
    num_timesteps: int
    eval_freq: int
    d_max: float
    beta: float
    kappa: float
    discount: float
    epsilon_start: float
    epsilon_end: float
    exploration_fraction: float
    prob_slip: float

    resume: bool
    render_eval: bool
    learning_rate: Optional[float] = None
    use_lr_schedule: bool = False


DEFAULT_CONFIGS = {
    "GridMap01": {
        "num_timesteps": 500 * 25,
        "d_max": 12,
        "eval_freq": 5,
        "use_lr_schedule": True,
    },
    "GridMap02": {
        "num_timesteps": 10000 * 15,
        "d_max": 8,
        "eval_freq": 10,
        "use_lr_schedule": True,
    },
    "GridMap03": {
        "num_timesteps": 10000 * 250,
        "d_max": 50,
        "eval_freq": 10,
        "learning_rate": 0.5,
    },
    "GridMap04": {
        "num_timesteps": 10000 * 250,
        "d_max": 50,
        "eval_freq": 10,
        "learning_rate": 0.25,
    },
}


def get_config() -> Config:
    parser = argparse.ArgumentParser(
        description="Run an experiment with the given settings and hyperparameters"
    )
    parser.add_argument(
        "--env",
        required=True,
        help="Which env to run?",
        type=str,
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["sparse", "symbolic", "lavaei2020"],
        type=str,
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for the experiments"
    )
    parser.add_argument(
        "--log-dir",
        type=lambda p: Path(p).absolute(),
        default=Path.cwd() / "logs",
        help="Directory to store logs",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        help="Total number of episodes to run the training for",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        help="Frequency (in number of episodes) to run evaluations",
    )
    lr_group = parser.add_mutually_exclusive_group()
    lr_group.add_argument("--lr", type=float, default=0.25, help="Learning rate")
    lr_group.add_argument(
        "--lr-schedule", action="store_true", help="Use the default LR schedule"
    )
    parser.add_argument(
        "--dmax", type=float, help="d_max hyperparameter for symbolic automaton"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=50,
        help="Beta hyperparameter for tau-MDP",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.5,
        help="Kappa hyperparameter for lavaei2020",
    )
    parser.add_argument("--discount", type=float, default=1.0, help="Discount rate")
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon for exploration",
    )
    parser.add_argument(
        "--exploration-end-fraction",
        type=float,
        default=0.5,
        help="The fraction of iterations to be completed before stopping epsilon-greedy",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Lower bound epsilon for exploration",
    )

    parser.add_argument("--prob-slip", type=float, default=0.1)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training using the latest checkpoint?",
    )

    parser.add_argument(
        "--render-eval", action="store_true", help="Render evaluation episodes"
    )

    args = parser.parse_args()

    env_str: str = args.env
    default = None
    for k, v in DEFAULT_CONFIGS.items():
        if env_str.startswith(k):
            default = v
            break
    assert default is not None, "Unknown env given {env_str}"

    return Config(
        method=args.method,
        env=env_str,
        num_timesteps=args.num_timesteps or default["num_timesteps"],
        eval_freq=args.eval_freq or default["eval_freq"],
        log_dir=args.log_dir,
        seed=args.seed,
        learning_rate=args.lr or default.get("learning_rate"),
        use_lr_schedule=args.lr_schedule or default.get("use_lr_schedule", False),
        d_max=args.dmax or default["d_max"],
        beta=args.beta,
        kappa=args.kappa,
        discount=args.discount,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        exploration_fraction=args.exploration_end_fraction,
        prob_slip=args.prob_slip,
        resume=args.resume,
        render_eval=args.render_eval,
    )


def configure(folder: Path, format_strings: List[str], log_suffix: str) -> Logger:
    """
    Configure the current
    :param folder: the save location
        (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
    :param format_strings: the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    :return: The logger object.
    """
    folder.mkdir(parents=True, exist_ok=True)
    output_formats = [
        make_output_format(f, str(folder), log_suffix) for f in format_strings
    ]

    logger = Logger(folder=str(folder), output_formats=output_formats)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.info(f"Logging to {str(folder)}")
    return logger


class _LRSchedule:
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, progress: float) -> float:
        # Progress goes from 1.0 -> 0.0
        n_iter = self.max_iterations * (1.0 - progress)
        return max(0.01, 1 / (n_iter + 1))


def main():
    config = get_config()
    log_dir = config.log_dir / f"{config.env}" / f"{config.method}"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    seed_str = f"{config.seed:05d}" if config.seed is not None else "None"
    LOG = configure(log_dir, ["stdout", "log", "csv", "tensorboard"], seed_str)
    env = gym.make(
        config.env,
        seed=config.seed,
        reward_method=config.method,
        kappa=config.kappa,
        beta=config.beta,
        d_max=config.d_max,
        prob_of_slip=config.prob_slip,
    )
    LOG.info(f"Initializing Q-learner")

    if config.use_lr_schedule:
        learning_rate = _LRSchedule(config.num_timesteps)
    else:
        assert config.learning_rate is not None
        learning_rate = config.learning_rate
    controller = TabularQLearning(
        env=env,
        learning_rate=learning_rate,
        gamma=config.discount,
        exploration_initial_eps=config.epsilon_start,
        exploration_final_eps=config.epsilon_end,
        exploration_fraction=config.exploration_fraction,
        seed=config.seed,
    )
    controller.set_logger(LOG)

    eval_render_mode = "human" if config.render_eval else None
    eval_env = gym.make(
        config.env,
        seed=config.seed,
        reward_method=config.method,
        kappa=config.kappa,
        beta=config.beta,
        d_max=config.d_max,
        prob_of_slip=config.prob_slip,
        render_mode=eval_render_mode,
    )

    controller.learn(
        total_timesteps=config.num_timesteps,
        eval_log_path=str(log_dir),
        eval_env=eval_env,
        eval_freq=config.eval_freq,
        resume=config.resume,
    )


if __name__ == "__main__":
    main()
