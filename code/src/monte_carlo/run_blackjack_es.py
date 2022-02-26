# Greg Attra
# 02/24/2022

'''
Executable for running MC eval on blackjack
'''

import gym
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, Tuple
from lib.monte_carlo.blackjack import space_dims
from lib.monte_carlo.policy import create_blackjack_policy
from lib.monte_carlo.blackjack import first_visit_mc, first_visit_mc_es

BLACKJACK_GAME_TAG = 'Blackjack-v1'

def dict_to_array(d: Dict, shape: Tuple) -> np.ndarray:
    '''
    Convert a dictionary to a numpy array for visualization
    '''
    arr = np.zeros(shape)
    for key, value in d.items():
        # key = eval(key)
        idx = tuple([int(_) for _ in key])
        arr[idx] = value
    return arr


def main():
    env = gym.make(BLACKJACK_GAME_TAG)
    Q_10k, policy = first_visit_mc_es(env, 1., 500000)
    obs_dims = space_dims(env.observation_space)
    dims = obs_dims + (env.action_space.n,)
    Q_10k = dict_to_array(Q_10k, dims)

    f, axs = plt.subplots(2, 2)
    axs[0, 0].matshow(np.average(Q_10k[:, :, 0], axis=2))
    axs[0, 0].set_xlim([0.5, 10.5])
    axs[0, 0].set_xlabel('Dealer Showing (1 = Ace)')
    axs[0, 0].set_ylim([11.5, 21.5])
    axs[0, 0].set_ylabel('Player Sum')
    axs[0, 0].set_title('10K - No Usable Ace')
    axs[0, 0].xaxis.tick_bottom()

    axs[1, 0].matshow(np.average(Q_10k[:, :, 1], axis=2))
    axs[1, 0].set_xlim([0.5, 10.5])
    axs[1, 0].set_xlabel('Dealer Showing (1 = Ace)')
    axs[1, 0].set_ylim([11.5, 21.5])
    axs[1, 0].set_ylabel('Player Sum')
    axs[1, 0].set_title('10K - Usable Ace')
    axs[1, 0].xaxis.tick_bottom()

    axs[0, 1].matshow(np.argmax(Q_10k[:, :, 0], axis=2))
    axs[0, 1].set_xlim([0.5, 10.5])
    axs[0, 1].set_xlabel('Dealer Showing (1 = Ace)')
    axs[0, 1].set_ylim([11.5, 21.5])
    axs[0, 1].set_ylabel('Player Sum')
    axs[0, 1].set_title('500K Episodes - No Usable Ace')
    axs[0, 1].xaxis.tick_bottom()

    axs[1, 1].matshow(np.argmax(Q_10k[:, :, 1], axis=2))
    axs[1, 1].set_xlim([0.5, 10.5])
    axs[1, 1].set_xlabel('Dealer Showing (1 = Ace)')
    axs[1, 1].set_ylim([11.5, 21.5])
    axs[1, 1].set_ylabel('Player Sum')
    axs[1, 1].set_title('500K Episodes - Usable Ace')
    axs[1, 1].xaxis.tick_bottom()
    
    plt.show()


if __name__ == '__main__':
    main()
