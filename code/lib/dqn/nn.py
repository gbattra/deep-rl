# Greg Attra
# 03.29.22

'''
Functions for non-linear function approximation with neural networks.
Come code inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

from typing import Callable
from matplotlib import dviread
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset

from lib.dqn.buffer import ReplayBuffer, Transition


class NeuralNetwork(nn.Module):
    def __init__(
            self,
            input_size: int,
            l1_dim: int,
            l2_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, l1_dim),
            nn.ReLU(),
            nn.Linear(l1_dim, l2_dim),
            nn.ReLU(),
            nn.Linear(l2_dim, 1)
        )

    def forward(self, x):
        return self.network(x)


class Dqn(nn.Module):
    def __init__(
            self,
            network: nn.Sequential) -> None:
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network(x)


def optimize_dqn(
        policy_net: Dqn,
        target_net: Dqn,
        buffer: ReplayBuffer,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        batch_size: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(buffer) < batch_size:
        return

    sample = buffer.sample(batch_size)
    batch = Transition(*zip(*sample))

    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)

    non_terminal_states = torch.cat([s for s in batch.next_state if s is not None])
    non_terminal_map = map(lambda state: state is not None, batch.next_state)
    non_terminal_mask = torch.tensor(tuple(non_terminal_map), device=device, dtype=torch.bool)

    next_state_q_vals = torch.zeros(batch_size, device=device)
    print(non_terminal_states)
    next_state_q_vals[non_terminal_mask] = target_net(non_terminal_states).max(1)[0].detach()
    
    target_q_vals = ((next_state_q_vals * gamma) + rewards)
    est_q_vals = policy_net(states).gather(1, actions)

    loss = loss_fn(est_q_vals, target_q_vals.unsqueeze(1))
    # print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
