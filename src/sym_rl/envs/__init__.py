from gym.envs.registration import register


def _custom_register(id: str, **kwargs):
    register(
        id=id,
        kwargs={
            "reward_method": "sparse",
            "kappa": 0.5,
            "beta": 50,
            # "d_max": 50,
            "prob_of_slip": 0.1,
            "penalize_rejecting": False,
            "reward_scaling": 1.0,
        },
        **kwargs,
    )


_custom_register(
    id="GridMap01UnboundedReach-v0",
    entry_point="sym_rl.envs.gridworld.map01:GridWorldMap01_UnboundedReach",
    max_episode_steps=25,
)

_custom_register(
    id="GridMap01BoundedReach-v0",
    entry_point="sym_rl.envs.gridworld.map01:GridWorldMap01_BoundedReach",
    max_episode_steps=25,
)

_custom_register(
    id="GridMap02BoundedRecurrence1-v0",
    entry_point="sym_rl.envs.gridworld.map02:GridWorldMap02_BoundedRecurrence1",
    max_episode_steps=15,
)

_custom_register(
    id="GridMap02BoundedRecurrence2-v0",
    entry_point="sym_rl.envs.gridworld.map02:GridWorldMap02_BoundedRecurrence2",
    max_episode_steps=15,
)

_custom_register(
    id="GridMap03Sequential-v0",
    entry_point="sym_rl.envs.gridworld.map03:GridWorldMap03_Sequential",
    max_episode_steps=250,
)

_custom_register(
    id="GridMap03BoundedSequential-v0",
    entry_point="sym_rl.envs.gridworld.map03:GridWorldMap03_BoundedSequential",
    max_episode_steps=250,
)

_custom_register(
    id="GridMap04BranchReach-v0",
    entry_point="sym_rl.envs.gridworld.map03:GridWorldMap03_Sequential",
    max_episode_steps=250,
)
