# Greg Attra
# 03/25/2022

'''
Implemenation of the four rooms domain
'''

from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple
from lib.function_approximation.features import all_features
import gym
import numpy as np


class FourRoomsAction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Slip(IntEnum):
    NONE = 0
    NEGATIVE = 1
    POSITIVE = 2


class FourRooms(gym.Env):
    WALL: int = -1
    GOAL: int = 1
    EMPTY: int = 0
    
    def __init__(
            self,
            arena: np.ndarray,
            noise: float = .0) -> None:
        super().__init__()
        self.observation_size = 1
        self.arena = arena
        self.noise = noise
        self.action_space = gym.spaces.Discrete(len(FourRoomsAction))
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Discrete(self.arena.shape[0]),
            gym.spaces.Discrete(self.arena.shape[1])
        ])
        self.pos = (self.arena.shape[0]-1, 0)

    def reset(self) -> Tuple[int, int]:
        self.pos = (self.arena.shape[0]-1, 0)
        return [self._pos_idx(self.pos)]

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _action_to_dydx(self, action: FourRoomsAction) -> Tuple[int, int]:
        action_map = {
            FourRoomsAction.UP: (-1, 0),
            FourRoomsAction.RIGHT: (0, 1),
            FourRoomsAction.DOWN: (1, 0),
            FourRoomsAction.LEFT: (0, -1)
        }
        return action_map[action]

    def _clamp_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        y, x = pos
        if y < 0:
            y = 0
        if y >= self.arena.shape[0]:
            y = self.arena.shape[0] - 1
        if x < 0:
            x = 0
        if x >= self.arena.shape[1]:
            x = self.arena.shape[1] - 1
        return (y, x)
    
    def _slip_action(self, action: FourRoomsAction, slip: Slip):
        """
        Apply a chosen slip type to an action
        """
        new_action_idx = action.value

        if slip == Slip.NONE:
            return FourRoomsAction(new_action_idx)

        if slip == Slip.NEGATIVE:
            new_action_idx -= 1
            if new_action_idx < 0:
                new_action_idx = len(FourRoomsAction) - 1
        if slip == Slip.POSITIVE:
            new_action_idx += 1
            if new_action_idx >= len(FourRoomsAction):
                new_action_idx = 0

        action_taken = FourRoomsAction(new_action_idx)
        return action_taken

    def _generate_slip(self, noise: float) -> Slip:
        """
        Generate a slip type via probability distribution function
        over all slip types
        """
        slip_pdf = [1. - noise, (noise / 2.), (noise / 2.)]
        x = self.np_random.random()
        a = 0
        while x > sum(slip_pdf[:a+1]):
            a += 1
        slip = Slip(a)
        return slip

    def _domain_slip(self, action: FourRoomsAction) -> FourRoomsAction:
        """
        Apply domain noise to an action. Choose a type of
        slip from a probability distribution function of
        all types of slips.
        """
        slip = self._generate_slip(self.noise)
        action_taken = self._slip_action(action, slip)
        return action_taken

    def _take_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        action = self._domain_slip(FourRoomsAction(action))
        dy, dx = self._action_to_dydx(action)
        pos = (pos[0] + dy, pos[1] + dx)
        pos = self._clamp_pos(pos)
        return pos

    def _pos_idx(self, pos: Tuple[int, int]) -> int:
        return (pos[0] * self.arena.shape[1]) + pos[1]

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        new_pos = self._take_action(self.pos, action)
        done = False
        rwd = 0
        if self.arena[new_pos] == self.WALL:
            new_pos = self.pos
        if self.arena[new_pos] == self.GOAL:
            done = True
            rwd = 1
        self.pos = new_pos
        pos_idx = self._pos_idx(self.pos)
        return [pos_idx], rwd, done, {}


class FourRoomsOneHot(FourRooms):
    def __init__(self, arena: np.ndarray, noise: float = 0) -> None:
        super().__init__(arena, noise)
        self.observation_size = arena.shape[0] * self.arena.shape[1]

    def reset(self) -> Tuple[int, int]:
        s = super().reset()[0]
        s_enc = [.0] * self.observation_size
        s_enc[s] = 1.
        return s_enc

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        s_prime, rwd, done, results = super().step(action)
        s_prime_enc = [.0] * self.observation_size
        s_prime_enc[s_prime[0]] = 1.
        return s_prime_enc, rwd, done, results
        

class FourRoomsCoords(FourRooms):
    def __init__(self, arena: np.ndarray, noise: float = 0) -> None:
        super().__init__(arena, noise)
        self.observation_size = 2

    def reset(self) -> Tuple[int, int]:
        _ = super().reset()
        return self.pos

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        _, rwd, done, results = super().step(action)
        return self.pos, rwd, done, results


class FourRoomsArena(FourRooms):
    PLAYER: int = 2

    def __init__(self, arena: np.ndarray, noise: float = 0) -> None:
        super().__init__(arena, noise)
        self.observation_size = arena.shape[0] * arena.shape[1]

    def reset(self) -> np.ndarray:
        _ = super().reset()
        mapp = self.arena.copy()
        mapp[self.pos[0], self.pos[1]] = self.PLAYER
        return list(mapp.flatten())

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        _, rwd, done, results = super().step(action)

        mapp = self.arena.copy()
        mapp[self.pos[0], self.pos[1]] = self.PLAYER

        return list(mapp.flatten()), rwd, done, results


def goal() -> Tuple[int, int]:
    return (0, 10)


def doors() -> List[Tuple[int, int]]:
    return [
        (5, 1),
        (2, 5),
        (6, 8),
        (9, 5)
    ]


def four_rooms_arena() -> np.ndarray:
    arena = np.ones((11, 11)) * FourRooms.EMPTY
    arena[5, 0:6] = FourRooms.WALL
    arena[6, 5:11] = FourRooms.WALL
    arena[:, 5] = FourRooms.WALL
    for door in doors():
        arena[door[0], door[1]] = FourRooms.EMPTY
    g = goal()
    arena[g[0], g[1]] = FourRooms.GOAL

    return arena
