# Greg Attra
# 03.29.22

'''
Executable for training simple NN
'''

import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Callable, Tuple
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from lib.dqn.nn import NeuralNetwork


N_SAMPLES: int = 500
N_EPOCHS: int = 500
BATCH_SIZE: int = 32


def generate_data() -> Tuple[Tensor, Tensor]:
    X = torch.Tensor(np.linspace(-10, 10, N_SAMPLES))
    Y = torch.Tensor(np.array([1 + x**2 for x in X]))

    return X, Y

def generate_dataset() -> TensorDataset:
    X, Y = generate_data()
    D = TensorDataset(X.reshape(-1, 1), Y.reshape(-1, 1))
    return D


def get_train_test_dataloaders(dataset: TensorDataset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
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
        loss_fn: Callable) -> float:
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
    return test_loss


def main():
    dims = [8, 16, 64, 128]
    colors = [
        (1., .0, .0),
        (.0, 1., .0),
        (.0, .0, 1.),
        (1., .0, 1.)
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = generate_dataset()
    train_dataloader, test_dataloader = get_train_test_dataloaders(dataset, BATCH_SIZE)
    
    X, Y = generate_data()
    plt.plot(Y, X, color=(0.1, 0.1, 0.1), alpha=0.5, label='True Function')

    for dim, color in zip(dims, colors):
        for lr, style in [(0.1, 'solid'), (0.01, 'dotted')]:
            model = NeuralNetwork(1, dim, dim).to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            total_results = np.zeros((N_EPOCHS))
            for e in range(N_EPOCHS):
                print(f'-------------- Epoch {e} --------------')
                train(model, loss_fn, train_dataloader, optimizer)
                test_loss = test(model, test_dataloader, loss_fn)
                total_results[e] = test_loss
            # plt.plot(total_results, color=color, linestyle=style, label=f'Dim Size: {dim} | LR: {lr}')

            y_hats = model(X.unsqueeze(1).to(device)).cpu().detach().numpy()
            plt.plot(y_hats, X.detach().numpy(), color=color, label=f'Dim Size: {dim} | LR: {lr}', linestyle=style)
    # plt.ylabel('Avg. Loss')
    # plt.xlabel('Epoch')
    plt.title('Learned Functions')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()