import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Logger, make_output_format
from stable_baselines3.dqn.dqn import DQN

import sym_rl.envs  # pyright: reportUnusedImport=false


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
    render_eval: bool
    learning_rate: Optional[float] = None

    n_procs: int = 1


EPISODE_LENGTH = 1500


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
        default=EPISODE_LENGTH * 25000,
        help="Total number of episodes to run the training for",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=250,
        help="Frequency (in number of episodes) to run evaluations",
    )
    lr_group = parser.add_mutually_exclusive_group()
    lr_group.add_argument("--lr", type=float, help="Learning rate")
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

    parser.add_argument(
        "--render-eval", action="store_true", help="Render evaluation episodes"
    )
    parser.add_argument(
        "--n-procs", type=int, default=1, help="Number of environments to spawn."
    )

    args = parser.parse_args()

    env_str: str = args.env

    return Config(
        method=args.method,
        env=env_str,
        num_timesteps=args.num_timesteps,
        eval_freq=args.eval_freq,
        log_dir=args.log_dir,
        seed=args.seed,
        d_max=args.dmax,
        beta=args.beta,
        kappa=args.kappa,
        discount=args.discount,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        exploration_fraction=args.exploration_end_fraction,
        render_eval=args.render_eval,
        learning_rate=args.lr,
        n_procs=args.n_procs,
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


def main():
    config = get_config()
    seed_str = f"{config.seed:05d}" if config.seed is not None else "None"

    log_dir = config.log_dir / f"{config.env}" / f"{config.method}" / seed_str
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_dir / f"chk"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = log_dir / f"monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    eval_freq_episodes = max(config.eval_freq // config.n_procs, 1)
    eval_freq = EPISODE_LENGTH * eval_freq_episodes
    log_interval = 10 # episodes
    checkpoint_freq = EPISODE_LENGTH * 10  # timesteps

    LOG = configure(log_dir, ["stdout", "log", "csv", "tensorboard"], seed_str)
    LOG.info(f"Logging to {log_dir}")
    LOG.info(f"Creating {config.env} environment")
    LOG.info(f"Creating {config.n_procs} copies")

    env_kwargs = dict(
        reward_method=config.method,
        kappa=config.kappa,
        beta=config.beta,
    )
    env = make_vec_env(
        config.env,
        n_envs=config.n_procs,
        seed=config.seed,
        monitor_dir=str(monitor_dir),
        env_kwargs=env_kwargs,
    )
    LOG.info(f"Initializing DQN")

    controller = DQN(
        policy="MultiInputPolicy",
        env=env,
        gamma=config.discount,
        exploration_initial_eps=config.epsilon_start,
        exploration_final_eps=config.epsilon_end,
        exploration_fraction=config.exploration_fraction,
        seed=config.seed,
    )
    controller.set_logger(LOG)

    eval_render_mode = "human" if config.render_eval else None
    eval_env = make_vec_env(
        config.env,
        n_envs=max(config.n_procs // 2, 1),
        seed=config.seed,
        env_kwargs=dict(
            **env_kwargs,
            render_mode=eval_render_mode,
        ),
    )

    LOG.info(
        f"Saving checkpoints to: {checkpoint_dir} (Frequency: {checkpoint_freq} timesteps)"
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="model",
    )

    LOG.info(f"Begin training DQN Model")
    LOG.info(f"Will evaluate every {eval_freq} timesteps")
    LOG.info(f"Will log every {log_interval} episodes")
    controller.learn(
        total_timesteps=config.num_timesteps,
        eval_log_path=str(log_dir),
        eval_env=eval_env,
        eval_freq=eval_freq,
        callback=checkpoint_callback,
        log_interval=log_interval,
    )


if __name__ == "__main__":
    main()
