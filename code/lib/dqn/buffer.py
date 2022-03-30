# Greg Attra
# 03.30.22

'''
Buffer functions. Some code inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

from collections import deque
import random
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple

from torch import Tensor
import torch
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class Transition:
    state: Tensor
    action: Tensor
    reward: Tensor


class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.buffer = deque([], maxlen=size)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def buffer_to_dataloader(buffer: ReplayBuffer, batch_size: int) -> DataLoader:
    pass
