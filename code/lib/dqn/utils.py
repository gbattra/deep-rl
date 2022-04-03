# Greg Attra
# 04.03.22

'''
Plot episode durations in real time. Borrowed from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

from typing import List
import matplotlib.pyplot as plt
import torch
from IPython import display

def plot_durations(durations: List[int]):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    display.clear_output(wait=True)
    display.display(plt.gcf())