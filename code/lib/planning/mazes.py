# Greg Attra
# 03.12.2022

'''
Domain implementation for blocking and shortcut mazes
'''

from enum import IntEnum
from typing import Any, Dict, Tuple
from gym import spaces, Env
import numpy as np


class MazeAction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Maze(Env):
    START: Tuple[int, int] = (5, 3)
    GOAL: Tuple[int, int] = (0, 8)
    WALL: int = -1

    def __init__(
            self,
            start_maze: np.ndarray,
            end_maze: np.ndarray,
            toggle_step: int) -> None:
        super().__init__()
        self.curr_maze = end_maze
        self.toggled = False
        self.start_maze = start_maze
        self.end_maze = end_maze
        self.toggle_step = toggle_step
        self.pos = self.START
        self.action_space = spaces.Discrete(len(MazeAction))
        self.observation_space = spaces.Tuple([
            spaces.Discrete(self.start_maze.shape[0]),
            spaces.Discrete(self.start_maze.shape[1])
        ])
        self.t = 0

    def _action_to_dydx(self, action: MazeAction) -> Tuple[int, int]:
        action_map = {
            MazeAction.UP: (-1, 0),
            MazeAction.RIGHT: (0, 1),
            MazeAction.DOWN: (1, 0),
            MazeAction.LEFT: (0, -1)
        }
        return action_map[action]

    def _clamp_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        y, x = pos
        if y < 0:
            y = 0
        if y >= self.curr_maze.shape[0]:
            y = self.curr_maze.shape[0] - 1
        if x < 0:
            x = 0
        if x >= self.curr_maze.shape[1]:
            x = self.curr_maze.shape[1] - 1
        return (y, x)

    def _take_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        dy, dx = self._action_to_dydx(MazeAction(action))
        pos = (pos[0] + dy, pos[1] + dx)
        pos = self._clamp_pos(pos)
        return pos

    def reset(self) -> Tuple[int, int]:
        self.toggled = False
        self.pos = self.START
        self.t = 0
        self.curr_maze = self.start_maze
        return self.pos

    def step(
            self,
            action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        self.t += 1
        if self.t >= self.toggle_step and not self.toggled:
            self.toggled = True
            self.curr_maze = self.end_maze
        new_pos = self._take_action(self.pos, action)
        done = False
        rwd = 0
        if self.curr_maze[new_pos] == self.WALL:
            new_pos = self.pos
        if new_pos == self.GOAL:
            new_pos = self.START
            done = True
            rwd = 1
        self.pos = new_pos
        return self.pos, rwd, done, {}


def blocking_maze_start() -> np.ndarray:
    maze = np.zeros((6, 9))
    maze[3, :8] = Maze.WALL
    print('Start Maze:')
    print(maze)
    return maze


def blocking_maze_end() -> np.ndarray:
    maze = np.zeros((6, 9))
    maze[3, 1:9] = Maze.WALL
    print('End Maze:')
    print(maze)
    return maze
