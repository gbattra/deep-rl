# Greg Attra
# 03/05/2021

'''
Implementation of the windy gridworld domain
'''

from enum import IntEnum
from typing import Any, Dict, Tuple
from gym import Env, spaces
import numpy as np
from tqdm import trange
from lib.monte_carlo.algorithms import on_policy_mc_control_epsilon_soft
from lib.temporal_difference.algorithms import expected_sarsa, sarsa

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class WindyGridworld(Env):
    GOAL = 1

    def __init__(self, wind_grid: np.ndarray) -> None:
        super().__init__()
        self.wind_grid = wind_grid
        self.pos = (0, 0)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple([
            spaces.Discrete(self.wind_grid.shape[0]),
            spaces.Discrete(self.wind_grid.shape[1])
        ])

    def action_to_dydx(self, action: Action) -> Tuple[int, int]:
        action_map = {
            Action.UP: (-1, 0),
            Action.RIGHT: (0, 1),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1)
        }
        return action_map[action]

    def clamp_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        y, x = pos
        if y < 0:
            y = 0
        if y >= self.wind_grid.shape[0]:
            y = self.wind_grid.shape[0] - 1
        if x < 0:
            x = 0
        if x >= self.wind_grid.shape[1]:
            x = self.wind_grid.shape[1] - 1
        return (y, x)

    def take_action(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        dy, dx = self.action_to_dydx(Action(action))
        pos = (pos[0] + dy, pos[1] + dx)
        pos = self.clamp_pos(pos)
        return pos

    def apply_wind(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        wind = self.wind_grid[pos]
        pos = (pos[0] + int(wind), pos[1])
        pos = self.clamp_pos(pos)
        return pos

    def reset(self) -> Tuple[int, int]:
        self.pos = (self.wind_grid.shape[0] // 2, 0)
        return self.pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        pos = self.take_action(self.pos, action)
        pos = self.apply_wind(pos)
        self.pos = pos
        rwd = -1.
        done = False
        if self.wind_grid[pos] == self.GOAL:
            rwd = .0
            done = True

        return (pos, rwd, done, {})


def windy_gridworld_1() -> np.ndarray:
    world = np.zeros([7, 10])
    world[:, 3:9] = -1
    world[:, 6:8] = -2
    world[3, 7] = WindyGridworld.GOAL
    print(world)
    return world


def monte_carlo_windy_gridworld(
    env: WindyGridworld,
    n_trials: int,
    n_episodes: int,
    gamma: float,
    eps: float) -> np.ndarray:
    trial_returns = np.zeros((n_trials, n_episodes))
    for t in trange(n_trials, desc='Trial'):
        returns = on_policy_mc_control_epsilon_soft(
            env,
            n_episodes,
            gamma,
            eps)
        trial_returns[t, :] = returns
    
    return trial_returns


def sarsa_windy_gridworld(
    env: WindyGridworld,
    n_trials: int,
    n_episodes: int,
    gamma: float,
    eps: float,
    alpha: float) -> np.ndarray:
    trial_returns = np.zeros((n_trials, n_episodes))
    for t in trange(n_trials, desc='Trial'):
        returns = sarsa(
            env,
            alpha,
            eps,
            gamma,
            n_episodes)
        trial_returns[t, :] = returns
    
    return trial_returns

def expected_sarsa_windy_gridworld(
    env: WindyGridworld,
    n_trials: int,
    n_episodes: int,
    gamma: float,
    eps: float,
    alpha: float) -> np.ndarray:
    trial_returns = np.zeros((n_trials, n_episodes))
    for t in trange(n_trials, desc='Trial'):
        returns = expected_sarsa(
            env,
            alpha,
            eps,
            gamma,
            n_episodes)
        trial_returns[t, :] = returns
    
    return trial_returns