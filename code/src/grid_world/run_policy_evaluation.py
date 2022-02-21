#!/usr/bin/env python

# Greg Attra
# 02/12/2022

# Driver class for running policy evaluation in the grid world problem

import numpy as np
import matplotlib.pyplot as plt

from lib.dynamic_programming.grid_world.grid_world import N_STATES, gw_dynamics, Action
from lib.dynamic_programming.policy_iteration import policy_evaluation


def main():
    policy = np.ones((N_STATES, len(Action))) / float(len(Action))
    dynamics = gw_dynamics()
    values = policy_evaluation(policy, dynamics, N_STATES, len(Action)).reshape(
                int(np.sqrt(N_STATES)),
                int(np.sqrt(N_STATES)))
    
    plt.matshow(values)
    for (i, j), z in np.ndenumerate(values):
        plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()
