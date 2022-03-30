# Greg Attra
# 03.29.22

'''
Functions for non-linear function approximation with neural networks
'''

from torch import nn


INPUT_SIZE: int = 500


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
        self.network = nn.Sequential

    def forward(self, x):
        return self.network(x)


def simple_dqn_network(input_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential([
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, output_size)
    ])