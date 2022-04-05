# Greg Attra
# 03.30.22

'''
Functions for policy generation / action selection
(some code inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
'''

import math
from typing import Callable
from lib.dqn.nn import Dqn

import numpy as np
import torch


def generate_dqn_policy(
        q_net: Dqn,
        n_actions: int,
        epsilon_start: float = 1.,
        epsilon_end: float = .01,
        epsilon_decay: float = .999) -> Callable[[torch.Tensor], torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def select_action(X: np.ndarray, t: int):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** t))
        if np.random.random() < epsilon:
            return torch.tensor([[np.random.choice(n_actions)]], device=device).item()

        with torch.no_grad():
            return q_net(torch.Tensor(X).to(device).float()).argmax().item()

    return select_action
