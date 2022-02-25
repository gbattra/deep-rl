# Greg Attra
# 02/24/2022

'''
Executable for running MC eval on blackjack
'''

import gym
import matplotlib.pyplot as plt
import numpy as np

from lib.monte_carlo.policy import create_blackjack_policy
from lib.monte_carlo.blackjack import first_visit_mc_eval

BLACKJACK_GAME_TAG = 'Blackjack-v1'

def main():
    env = gym.make(BLACKJACK_GAME_TAG)
    policy = create_blackjack_policy(dict())
    V_10k = first_visit_mc_eval(policy, env, 1., 10000)
    V_500k = first_visit_mc_eval(policy, env, 1., 500000)

    f, axs = plt.subplots(2, 2)
    axs[0, 0].matshow(V_10k[:, :, 0])
    axs[0, 0].set_xlim([0.5, 10.5])
    axs[0, 0].set_xlabel('Dealer Showing (1 = Ace)')
    axs[0, 0].set_ylim([11.5, 21.5])
    axs[0, 0].set_ylabel('Player Sum')
    axs[0, 0].set_title('10K - No Usable Ace')
    axs[0, 0].xaxis.tick_bottom()

    axs[1, 0].matshow(V_10k[:, :, 1])
    axs[1, 0].set_xlim([0.5, 10.5])
    axs[1, 0].set_xlabel('Dealer Showing (1 = Ace)')
    axs[1, 0].set_ylim([11.5, 21.5])
    axs[1, 0].set_ylabel('Player Sum')
    axs[1, 0].set_title('10K - Usable Ace')
    axs[1, 0].xaxis.tick_bottom()

    axs[0, 1].matshow(V_500k[:, :, 0])
    axs[0, 1].set_xlim([0.5, 10.5])
    axs[0, 1].set_xlabel('Dealer Showing (1 = Ace)')
    axs[0, 1].set_ylim([11.5, 21.5])
    axs[0, 1].set_ylabel('Player Sum')
    axs[0, 1].set_title('500K Episodes - No Usable Ace')
    axs[0, 1].xaxis.tick_bottom()

    axs[1, 1].matshow(V_500k[:, :, 1])
    axs[1, 1].set_xlim([0.5, 10.5])
    axs[1, 1].set_xlabel('Dealer Showing (1 = Ace)')
    axs[1, 1].set_ylim([11.5, 21.5])
    axs[1, 1].set_ylabel('Player Sum')
    axs[1, 1].set_title('500K Episodes - Usable Ace')
    axs[1, 1].xaxis.tick_bottom()
    
    plt.title('First-visit Monte Carlo Evaluation')
    plt.show()


if __name__ == '__main__':
    main()
