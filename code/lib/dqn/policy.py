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
        q_net: Dqn,
        n_actions: int,
        epsilon: float) -> Callable[[torch.Tensor], torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def select_action(X: torch.Tensor):
        if np.random.random() < epsilon:
            return torch.tensor([[np.random.choice(n_actions)]], device=device)

        with torch.no_grad():
            return torch.tensor([[q_net(X).argmax()]], device=device)

    return select_action
