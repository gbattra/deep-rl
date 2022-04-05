# Greg Attra
# 04.03.22

'''
Plot episode durations in real time. Borrowed from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

from typing import List
import matplotlib.pyplot as plt
from numpy import block
import torch
from IPython import display

def plot_durations(data: List[List[int]], labels: List[str]):
    plt.figure(2)
    plt.clf()
    
    plt.title('Training...')
    plt.xlabel('Episode')
    for d, label in zip(data, labels):
        d_t = torch.tensor(d, dtype=torch.float)
        plt.plot(d_t.numpy(), label=label)
        
        if len(d_t) >= 100:
            means = d_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label=label)
    plt.pause(0.001)
    plt.legend()
    plt.show(block=False)
