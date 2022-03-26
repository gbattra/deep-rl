# Greg Attra
# 03/25/22

'''
Functions for generating policies
'''

import numpy as np

from typing import Callable, List


def q_function(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    return x.dot(W)


def generate_policy(
        Q: Callable,
        epsilon: float) -> Callable[[np.ndarray], int]:

    def choose_action(s: np.ndarray, W: np.ndarray) -> int:
        a_vals = Q(s, W)
        if np.random.random() < epsilon:
            return np.random.randint(0, len(a_vals))
        return np.random.choice(np.flatnonzero(a_vals == a_vals.max()))
    
    return choose_action
