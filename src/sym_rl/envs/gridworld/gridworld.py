"""A GridWorld environment

A customized implementation of `GridWorldEnv` from
[`gym-examples`](https://github.com/Farama-Foundation/gym-examples)
"""

import enum
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import gym
import numpy as np
import pygame
from gym import spaces
from gym.utils.renderer import Renderer

WINDOW_MAX_WIDTH = 512
WINDOW_MAX_HEIGHT = 512


@enum.unique
class Action(enum.IntEnum):
    """List of actions that can be performed in the GridWorld.

    .. note:
        The actions are given these specific values so that for an action `a`, we have
        that `(a - 1) % 8` and `(a + 1) % 8` are adjacent to `a`.
    """

    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7
    STAY = 8


class GridWorldEnv(gym.Env):
    """A grid world MDP of some size with slippage issues.


    In this environment, if an agent takes an action to move along some direction, if
    that direction is unobstructed, the agent will move one cell along that direction
    with some probability (`1 - prob_of_slip`) or an orthogonal direction with
    probability (`prob_of_slip/2`) each. If the direction is obstructed, it stays in the
    same cell with probability `1 - prob_of_slip`.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        world_shape: Tuple[int, int],
        *,
        initial_pos: Tuple[int, int],
        prob_of_slip: float = 0.2,
        walls: Optional[Iterable[Tuple[int, int]]] = None,
        roi: Optional[Mapping[str, Iterable[Tuple[int, int]]]] = None,
        render_mode=None
    ):
        """
        Create a grid with some shape. You can specify some other interesting properties
        of the grid like regions of interest, non-labelled obstacles, slippage
        probability.

        Args:
            world_shape: (row x column) size of the grid
            initial_pos: Initial position (row, column) in the grid.
            prob_of_slip:
                Probability of slippage. Here, if an agent takes an action `a`, the
                probability of taking the action is `1 - prob_of_slip`, and the
                probability of taking an orthogonal direction is `prob_of_slip/2`.
            walls: A list of cells in the grid that need to be labelled as walls.
            roi: A labeled list of regions.
            unsafe_states: List of unsafe states
        """

        n_rows, n_cols = world_shape  # ny, nx
        self.observation_space = spaces.MultiDiscrete([n_cols, n_rows])
        self.action_space = spaces.Discrete(len(Action))

        self._shape = world_shape

        assert 0 <= initial_pos[0] < world_shape[0]
        assert 0 <= initial_pos[1] < world_shape[1]
        self._initial_pos = initial_pos
        self._agent_location = initial_pos

        assert 0 <= prob_of_slip < 1
        self._p_slip = prob_of_slip

        # A boolean grid such that a cell is `True` if it is an obstacle.
        self._obs_map = np.zeros(world_shape, dtype=bool)
        if walls is not None:
            for x, y in walls:
                self._obs_map[y, x] = True

        init_x, init_y = self._initial_pos
        assert self._obs_map[init_y, init_x] is not True

        # Regions of interest
        self._roi_labels: List[str] = []
        self._roi_map = np.full(world_shape, -1)  # Grid cell is -1 if unlabelled.
        if roi is not None:
            self._roi_labels = [key for key in roi.keys()]
            for i, key in enumerate(self._roi_labels):
                for x, y in roi[key]:
                    # Set the cell to the label index
                    self._roi_map[y, x] = i

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._renderer = Renderer(self.render_mode, self._render_frame)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        """Compute window size based on grid size"""
        # First, get the cell size in pixels based on grid shape
        if n_cols > n_rows:  # Wider than it is tall
            cell_size = WINDOW_MAX_WIDTH / n_cols
        else:
            cell_size = WINDOW_MAX_HEIGHT / n_rows
        # Then, compute the size of the window
        window_width = n_cols * cell_size
        window_height = n_rows * cell_size

        self.window_size = (window_width, window_height)
        self.cell_size = cell_size

    def _get_obs(self):
        y, x = self._agent_location
        return np.array([x, y])

    def _get_info(self):
        return {}

    def reset(self, seed=None, return_info=False, _=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self._initial_pos

        observation = self._get_obs()
        info = self._get_info()

        self._renderer.reset()
        self._renderer.render_step()

        return (observation, info) if return_info else observation

    def _increment_cell_using_action(
        self, state: Tuple[int, int], a: int
    ) -> Tuple[int, int]:
        if a == int(Action.STAY):
            return state

        row, col = state
        old_row, old_col = state

        nrow, ncol = self._shape
        a_enum = Action(a)

        move_north = a_enum in [Action.NORTH, Action.NORTHEAST, Action.NORTHWEST]
        move_south = a_enum in [Action.SOUTH, Action.SOUTHEAST, Action.SOUTHWEST]
        move_west = a_enum in [Action.WEST, Action.NORTHWEST, Action.SOUTHWEST]
        move_east = a_enum in [Action.EAST, Action.NORTHEAST, Action.SOUTHEAST]

        assert not (move_north and move_south)
        assert not (move_east and move_west)

        if move_north:
            row = min(row + 1, nrow - 1)
        if move_south:
            row = max(row - 1, 0)
        if move_east:
            col = min(col + 1, ncol - 1)
        if move_west:
            col = max(col - 1, 0)

        if self._occupied((row, col)):
            return (old_row, old_col)
        return (row, col)

    def step(self, action: int):
        a = Action(action)
        a_int = int(a)
        row, col = self._agent_location

        nrow, ncol = self._shape
        assert row < nrow and col < ncol

        if self._obs_map[row, col]:
            raise ValueError(
                "Given state {} is occupied by an static obstacle".format(
                    self._agent_location
                )
            )

        # HACK: Not useful unless we have a reward function
        reward = 0
        done = False

        if self._p_slip == 0:
            # Deterministic
            self._agent_location = self._increment_cell_using_action(
                self._agent_location, a
            )
        else:
            # Stochastic
            state = self._agent_location
            if a == Action.STAY:
                # Just stay
                forward = state
                left = state
                right = state
            else:
                # Get the next state and the two adjacent states
                # TODO: Is possible to move into a wall and slip in orthogonal directions.
                forward = self._increment_cell_using_action(state, a_int)
                left = self._increment_cell_using_action(state, (a_int - 1) % 8)
                right = self._increment_cell_using_action(state, (a_int + 1) % 8)

            choices = [forward, left, right]
            idx = self.np_random.choice(
                3, p=[1 - self._p_slip, self._p_slip / 2, self._p_slip / 2]
            )
            self._agent_location = choices[idx]

        observation = self._get_obs()
        info = self._get_info()

        self._renderer.render_step()

        return observation, reward, done, info

    def render(self):
        return self._renderer.get_renders()

    def _render_frame(self, mode):
        assert mode is not None

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        cell_size = self.cell_size  # The size of a single grid square in pixels
        width, height = self.window_size

        def to_pygame(coords: Tuple[float, float]):
            """Convert an object's coords into pygame coordinates (lower-left of object => top left in pygame coords)."""
            return (cell_size * coords[0], height - cell_size * (coords[1] + 1))

        # First we paint the target locations
        idx: Tuple[int, int]
        for idx in zip(*np.nonzero(self._roi_map >= 0)):  # type: ignore
            y, x = idx
            cell_location = to_pygame((x, y))
            pygame.draw.rect(
                canvas, (255, 0, 0), pygame.Rect(cell_location, (cell_size, cell_size))
            )
        # Now we draw the agent
        agent_y, agent_x = self._agent_location
        agent_location = to_pygame((agent_x + 0.5, agent_y - 0.5))  # Centroid of agent
        pygame.draw.circle(canvas, (0, 0, 255), agent_location, cell_size / 3)

        n_rows, n_cols = self._shape
        # Add vertical grid lines
        for x in range(n_cols + 1):
            pygame.draw.line(
                canvas,
                0,
                (cell_size * x, 0),
                (cell_size * x, height),
                width=3,
            )
        for y in range(n_rows + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, cell_size * y),
                (width, cell_size * y),
                width=3,
            )

        if mode == "human":
            assert self.window is not None
            assert self.clock is not None

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def unpack(observation, info: Dict) -> Mapping[str, int]:
        """Given a state tuple, get a dictionary mapping variables to value"""
        row, col = observation
        return {
            "x": col,
            "y": row,
        }

    def _occupied(self, state: Tuple[int, int]) -> bool:
        return self._obs_map[state]
