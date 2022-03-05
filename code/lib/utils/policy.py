# Greg Attra
# 03/05/022

'''
Policy functions
'''

from collections import defaultdict
from typing import Callable, Tuple

import numpy as np


def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # TODO
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        if np.random.random() < epsilon:
            action = np.random.randint(0, num_actions)
        else:
            action = np.random.choice(np.flatnonzero(Q[state] == Q[state].max()))

        return action

    return get_action


def policy_probability_e_greedy(
        actions: np.ndarray,
        epsilon: float) -> np.ndarray:
    action_probs = np.ones(actions.shape)
    max_idx = np.argmax(actions)
    action_probs *= epsilon
    action_probs[max_idx] = 1. - epsilon
    return action_probs
