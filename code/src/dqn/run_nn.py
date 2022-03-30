# Greg Attra
# 03.29.22

'''
Executable for training simple NN
'''

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from lib.dqn.nn import NeuralNetwork, get_train_test_dataloaders, test, train


N_EPOCHS: int = 500
BATCH_SIZE: int = 32

def main():
    dims = [8, 16, 64, 128]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for lr, style in [(0.1, 'solid'), (0.01, 'dotted')]:
        for dim in dims:
            model = NeuralNetwork(1, dim, dim).to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_dataloader, test_dataloader = get_train_test_dataloaders(BATCH_SIZE)

            total_results = np.zeros((N_EPOCHS))
            for e in range(N_EPOCHS):
                print(f'-------------- Epoch {e} --------------')
                train(model, loss_fn, train_dataloader, optimizer)
                test_loss = test(model, test_dataloader, loss_fn)
                total_results[e] = test_loss
            plt.plot(total_results, linestyle=style, label=f'Dim Size: {dim} | LR: {lr}')
    plt.ylabel('Avg. Loss')
    plt.xlabel('Epoch')
    plt.title('Simple NN')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()