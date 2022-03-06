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
from lib.temporal_difference.algorithms import expected_sarsa, n_step_sarsa, q_learning, sarsa


class GridworldAction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP_RIGHT = 4
    UP_LEFT = 5
    DOWN_RIGHT = 6
    DOWN_LEFT = 7
    NOOP = 8


class WindyGridworld(Env):
    GOAL = 1

    def __init__(
            self,
            wind_grid: np.ndarray,
            king_moves: bool = False,
            noop_action: bool = False,
            stochastic_wind: bool = False,
            max_t: int = 5000) -> None:
        super().__init__()
        self.wind_grid = wind_grid
        self.pos = (0, 0)
        self.t = 0
        self.max_t = max_t
        self.king_moves = king_moves
        self.noop_action = noop_action
        self.stochastic_wind = stochastic_wind
        self.action_space = spaces.Discrete(4 + (5 if king_moves and noop_action else (4 if king_moves else 0)))
        self.observation_space = spaces.Tuple([
            spaces.Discrete(self.wind_grid.shape[0]),
            spaces.Discrete(self.wind_grid.shape[1])
        ])
        print(self.to_string())

    def to_string(self) -> str:
        return f'Windy Gridworld - ' \
            + f'Action Space: {self.action_space.n} - ' \
            + f'King Moves: {str(self.king_moves)} - ' \
            + f'Noop Action: {str(self.noop_action)} - ' \
            + f'Stochastic Wind: {str(self.stochastic_wind)}'

    def action_to_dydx(self, action: GridworldAction) -> Tuple[int, int]:
        action_map = {
            GridworldAction.UP: (-1, 0),
            GridworldAction.RIGHT: (0, 1),
            GridworldAction.DOWN: (1, 0),
            GridworldAction.LEFT: (0, -1),
            GridworldAction.UP_RIGHT: (-1, 1),
            GridworldAction.UP_LEFT: (-1, -1),
            GridworldAction.DOWN_RIGHT: (1, 1),
            GridworldAction.DOWN_LEFT: (1, -1),
            GridworldAction.NOOP: (0, 0)
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

    def take_action(self, pos: Tuple[int, int], action: GridworldAction) -> Tuple[int, int]:
        dy, dx = self.action_to_dydx(GridworldAction(action))
        pos = (pos[0] + dy, pos[1] + dx)
        pos = self.clamp_pos(pos)
        return pos

    def apply_wind(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        sample = np.random.random()
        noise = 0
        if sample < 2./3.:
            noise = 1
        if sample < 1./3.:
            noise = -1
        wind = self.wind_grid[pos]
        wind += noise if self.stochastic_wind else 0
        pos = (pos[0] + int(wind), pos[1])
        pos = self.clamp_pos(pos)
        return pos

    def reset(self) -> Tuple[int, int]:
        self.pos = (self.wind_grid.shape[0] // 2, 0)
        self.t = 0
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
        if self.t >= self.max_t:
            done = True

        self.t += 1

        return (pos, rwd, done, {})


def windy_gridworld_1() -> np.ndarray:
    world = np.zeros([7, 10])
    world[:, 3:9] = -1
    world[:, 6:8] = -2
    world[3, 7] = WindyGridworld.GOAL
    return world


def monte_carlo_windy_gridworld(
        env: WindyGridworld,
        n_trials: int,
        n_episodes: int,
        gamma: float,
        eps: float) -> np.ndarray:
    trial_returns = np.zeros((n_trials, n_episodes))
    trial_episode_lengths = np.zeros((n_trials, n_episodes))
    for t in trange(n_trials, desc='Trial', leave=False):
        results = on_policy_mc_control_epsilon_soft(
            env,
            n_episodes,
            gamma,
            eps)
        returns = results['returns']
        episode_lengths = results['episode_lengths']
        trial_returns[t, :] = returns
        trial_episode_lengths[t, :] = episode_lengths
    
    return trial_returns, trial_episode_lengths


def sarsa_windy_gridworld(
        env: WindyGridworld,
        n_trials: int,
        n_episodes: int,
        gamma: float,
        eps: float,
        alpha: float) -> np.ndarray:
    trial_returns = np.zeros((n_trials, n_episodes))
    trial_episode_lengths = np.zeros((n_trials, n_episodes))
    for t in trange(n_trials, desc='Trial', leave=False):
        results = sarsa(
            env,
            alpha,
            eps,
            gamma,
            n_episodes)
        returns = results['returns']
        episode_lengths = results['episode_lengths']
        trial_returns[t, :] = returns
        trial_episode_lengths[t, :] = episode_lengths
    
    return trial_returns, trial_episode_lengths

def expected_sarsa_windy_gridworld(
        env: WindyGridworld,
        n_trials: int,
        n_episodes: int,
        gamma: float,
        eps: float,
        alpha: float) -> np.ndarray:
    trial_returns = np.zeros((n_trials, n_episodes))
    trial_episode_lengths = np.zeros((n_trials, n_episodes))
    for t in trange(n_trials, desc='Trial', leave=False):
        results = expected_sarsa(
            env,
            alpha,
            eps,
            gamma,
            n_episodes)
        returns = results['returns']
        episode_lengths = results['episode_lengths']
        trial_returns[t, :] = returns
        trial_episode_lengths[t, :] = episode_lengths
    
    return trial_returns, trial_episode_lengths


def n_step_sarsa_windy_gridworld(
        env: WindyGridworld,
        n_steps: int,
        n_trials: int,
        n_episodes: int,
        gamma: float,
        eps: float,
        alpha: float) -> np.ndarray:
    trial_returns = np.zeros((n_trials, n_episodes))
    trial_episode_lengths = np.zeros((n_trials, n_episodes))
    for t in trange(n_trials, desc='Trial', leave=False):
        results = n_step_sarsa(
            env,
            n_steps,
            alpha,
            eps,
            gamma,
            n_episodes)
        returns = results['returns']
        episode_lengths = results['episode_lengths']
        trial_returns[t, :] = returns
        trial_episode_lengths[t, :] = episode_lengths
    
    return trial_returns, trial_episode_lengths

def q_learning_windy_gridworld(
        env: WindyGridworld,
        n_trials: int,
        n_episodes: int,
        gamma: float,
        eps: float,
        alpha: float) -> np.ndarray:
    trial_returns = np.zeros((n_trials, n_episodes))
    trial_episode_lengths = np.zeros((n_trials, n_episodes))
    for t in trange(n_trials, desc='Trial', leave=False):
        results = q_learning(
            env,
            alpha,
            eps,
            gamma,
            n_episodes)
        returns = results['returns']
        episode_lengths = results['episode_lengths']
        trial_returns[t, :] = returns
        trial_episode_lengths[t, :] = episode_lengths
    
    return trial_returns, trial_episode_lengths
