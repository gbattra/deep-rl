# Greg Attra
# 03.29.22

'''
Executable for training simple NN
'''

from typing import Tuple
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from lib.dqn.nn import NeuralNetwork, get_train_test_dataloaders, test, train


LEARNING_RATE: float = 0.1
N_EPOCHS: int = 500
BATCH_SIZE: int = 32

def main():
    dims = [8, 16, 64, 128]
    lr = LEARNING_RATE

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dim in dims:
        model = NeuralNetwork(1, dim, dim).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_dataloader, test_dataloader = get_train_test_dataloaders(BATCH_SIZE)

        for e in range(N_EPOCHS):
            print(f'-------------- Epoch {e} --------------')
            train(model, loss_fn, train_dataloader, optimizer)
            test(model, test_dataloader, loss_fn)
        print('Done')


if __name__ == '__main__':
    main()