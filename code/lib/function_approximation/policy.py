# Greg Attra
# 03/25/22

'''
Functions for generating policies
'''

import numpy as np

from typing import Callable, List


def q_function(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    return x.dot(W)


def gradients(
        x: np.ndarray,
        W: np.ndarray,
        target: float,
        prediction: float,
        a: int) -> np.ndarray:
    pass


def generate_policy(
        Q: Callable,
        epsilon: float) -> Callable[[np.ndarray], int]:

    def choose_action(s: np.ndarray, W: np.ndarray) -> int:
        a_vals = Q(s, W)
        if np.random() < epsilon:
            return np.random.randint(0, len(a_vals))
        return np.argmax(a_vals)
    
    return choose_action
