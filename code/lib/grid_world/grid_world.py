# Greg Attra
# 02/13/2022

'''
Helper functions for setting up grid world dynamics / rewards
'''

from argparse import Action
from enum import Enum
from typing import Tuple
import numpy as np


N_STATES: int = 25

A_GOAL: Tuple[int, int] = (1, 4)
A_PRIME: Tuple[int, int] = (1, 0)
A_RWD: float = 10.

B_GOAL: Tuple[int, int] = (3, 4)
B_PRIME: Tuple[int, int] = (3, 2)
B_RWD: float = 5.

OOB_RWD: float = -1.


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def state_coords_to_id(x: int, y: int) -> int:
    '''
    Convert the x, y coords of the state to the
    int id of the state
    '''
    return int((y * np.sqrt(N_STATES)) + x)


def state_id_to_coords(s: int) -> Tuple[int, int]:
    '''
    Convert the id of the state to the x, y coord
    of the state cell
    '''
    y = s / np.sqrt(N_STATES)
    x = s % np.sqrt(N_STATES)

    return int(x), int(y)


def transition(s: int, a: Action) -> Tuple[int, float]:
    '''
    Get the next state when taking action a
    from state s
    '''
    x, y = state_id_to_coords(s)
    x_prime, y_prime = x, y
    rwd = 0

    if a == Action.UP:
        x_prime, y_prime = x, y+1
    if a == Action.DOWN:
        x_prime, y_prime =  x, y-1
    if a == Action.LEFT:
        x_prime, y_prime =  x-1, y
    if a == Action.RIGHT:
        x_prime, y_prime =  x+1, y
    
    if x_prime < 0 or x_prime > (np.sqrt(N_STATES) - 1):
        x_prime = x
        rwd = -1
    if y_prime < 0 or y_prime > (np.sqrt(N_STATES) - 1):
        y_prime = y
        rwd = -1

    return state_coords_to_id(x_prime, y_prime), rwd


def set_transport_dynamics(
        goal_state: int,
        prime_state: int,
        rwd: float,
        dynamics: np.ndarray) -> None:
    '''
    Override the dynamics for a goal state to transport
    the agent
    '''
    dynamics[goal_state, :, :, :] = 0
    dynamics[goal_state, :, prime_state, :] = 1, rwd


def gw_dynamics() -> np.ndarray:
    '''
    Build the dynamics function
    '''
    dynamics = np.zeros((N_STATES, len(Action), N_STATES, 2))
    for s in range(N_STATES):
        for a in range(len(Action)):
            s_prime, rwd = transition(s, a)
            dynamics[s, a, s_prime, :] = 1, rwd

    # override transition for goal states to transport agent
    set_transport_dynamics(
        state_coords_to_id(*A_GOAL),
        state_coords_to_id(*A_PRIME),
        A_RWD,
        dynamics)
    set_transport_dynamics(
        state_coords_to_id(*B_GOAL),
        state_coords_to_id(*B_PRIME),
        B_RWD,
        dynamics)
    
    return dynamics
