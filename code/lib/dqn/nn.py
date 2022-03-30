# Greg Attra
# 03.29.22

'''
Functions for non-linear function approximation with neural networks
'''

from dataclasses import replace
from typing import Callable, Tuple
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def generate_dataset() -> TensorDataset:
    X = torch.Tensor(np.linspace(-10, 10, INPUT_SIZE))
    Y = torch.Tensor(np.array([1 + x**2 for x in X]))
    D = TensorDataset(X.reshape(-1, 1), Y.reshape(-1, 1))
    return D


def get_train_test_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    dataset = generate_dataset()
    train_data = DataLoader(dataset[:400], batch_size=batch_size)
    test_data = DataLoader(dataset[400:], batch_size=batch_size)
    return train_data, test_data


# code inspired by https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def train(
        model: NeuralNetwork,
        loss_fn: Callable,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# code inspired by https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def test(
        model: NeuralNetwork,
        dataloader: DataLoader,
        loss_fn: Callable):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= float(num_batches)
    print(f"Test Avg loss: {test_loss} \r")