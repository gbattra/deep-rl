# Greg Attra
# 04.03.22

'''
Plot episode durations in real time. Borrowed from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

from typing import List
import matplotlib.pyplot as plt
import torch
from IPython import display

def plot_durations(durations: List[int], label: str):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.plot(durations_t.numpy(), label=label)

    plt.pause(0.001)
