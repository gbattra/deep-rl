# Greg Attra
# 01/22/2021

# Implementation of a simulation of the four room problem


import enum
from random import random
from typing import Tuple
import numpy as np


class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Slip(enum.Enum):
    NONE = 0
    NEGATIVE = 1
    POSITIVE = 2


class MapCell(enum.Enum):
    EMPTY = 0
    GOAL = 1
    WALL = 2


class State(Tuple[int, int]):
    def __init__(self) -> None:
        super().__init__()


class Transition:
    def __init__(self, state: State, reward: float) -> None:
        self.state: State = state
        self.reward: float = reward


class Simulation:
    def __init__(self, mapp: np.ndarray) -> None:
        self.map: np.ndarray = mapp

    def _valid_state(self, state: State) -> bool:
        """
        Determine if a given state is valid in the map
        """
        x_bound, y_bound = self.map.shape
        x, y = state
        if x < 0 or x >= x_bound:
            return False
        if y < 0 or y >= y_bound:
            return False
        
        return MapCell(self.map[x, y]) != MapCell.WALL

    def _get(self, state: State) -> MapCell:
        """
        Get the cell type (i.e. wall, goal, empty) at given state (x, y coords)
        """
        x, y = state
        return MapCell(self.map[x, y])

    def _update_state(self, state: State, action: Action) -> State:
        """
        Compute the new state when given action taken at given state
        """
        x, y = state

        if self._get((x, y)) == MapCell.GOAL:
            return (x, y)

        if action == Action.UP:
            return (x, y + 1)
        if action == Action.DOWN:
            return (x, y - 1)
        if action == Action.RIGHT:
            return (x + 1, y)
        if action == Action.LEFT:
            return (x - 1, y)

        return state

    def _slip_action(self, action: Action, slip: Slip):
        """
        Apply a chosen slip type to an action
        """
        new_action_idx = action.value

        if slip == Slip.NONE:
            return Action(new_action_idx)

        if slip == Slip.NEGATIVE:
            new_action_idx -= 1
            if new_action_idx < 0:
                new_action_idx = len(Action)
        if slip == Slip.POSITIVE:
            new_action_idx += 1
            if new_action_idx >= len(Action):
                new_action_idx = 0

        action_taken = Action(new_action_idx)
        if new_action_idx != action.value:
            print(f'Domain noise! Action taken: {str(action_taken)} ')
            
        return action_taken

    def _generate_slip(self) -> Slip:
        """
        Generate a slip type via probability distribution function
        over all slip types
        """
        # action pdf: .8  .1  .1
        action_cdf = [.8, .9, 1.]
        x = random()
        a = 0
        while x > action_cdf[a]:
            a += 1
        slip = Slip(a)
        return slip

    def _domain_slip(self, action: Action) -> Action:
        """
        Apply domain noise to an action. Choose a type of
        slip from a probability distribution function of
        all types of slips.
        """
        slip = self._generate_slip()
        action_taken = self._slip_action(action, slip)
        return action_taken

    def _state_reward(self, state: State) -> float:
        """
        Return the reward for a given state
        """
        if not self._valid_state(state):
            return .0
        
        x, y = state
        if self._get((x, y)) == MapCell.GOAL:
            return 1.0
        
        return 0
    
    def simulate(self, state: State, a: Action) -> Transition:
        """
        Simulate an action step:
            - augment chosen action with domain noise
            - compute new state upon taking action
            - if new state invalid, return current state and 0 reward
            - otherwise return reward for new state
        """
        action_taken: Action = self._domain_slip(a)
        new_state = self._update_state(state, action_taken)
        if not self._valid_state(new_state):
            return Transition(state, 0)
        reward = self._state_reward(new_state)
        return Transition(new_state, reward)

    def reset(self) -> State:
        """
        Reset the state of the agent to start state
        """
        return (0, 0)
