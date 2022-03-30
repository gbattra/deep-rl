# Greg Attra
# 03.30.22

'''
Functions for policy generation / action selection
(some code inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
'''

from typing import Callable
from lib.dqn.nn import Dqn

import numpy as np
import torch


def generate_dqn_policy(
        dqn: Dqn,
        n_actions: int,
        epsilon: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def select_action(X: torch.Tensor):
        if np.random.random() < epsilon:
            return torch.tensor([np.random.choice(n_actions)])

        with torch.no_grad():
            return dqn(X).max(1)[1]

    return select_action
