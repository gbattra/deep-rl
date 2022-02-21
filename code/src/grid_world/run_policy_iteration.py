#!/usr/bin/env python

# Greg Attra
# 02/12/2022

# Driver class for running policy iteration in the grid world problem

import numpy as np
import matplotlib.pyplot as plt

from lib.dynamic_programming.grid_world.grid_world import N_STATES, gw_dynamics, Action
from lib.dynamic_programming.policy_iteration import policy_iteration


def main():
    policy = np.ones((N_STATES, len(Action))) / float(len(Action))
    dynamics = gw_dynamics()
    policy = policy_iteration(policy, dynamics, N_STATES, len(Action))
    argmax_policy = np.argmax(policy, axis=1)

    readable_policy = []
    for a in argmax_policy:
        readable_policy.append(str(Action(a)).replace('Action.', ''))
    readable_policy = np.array(readable_policy)

    readable_policy = []
    for a in argmax_policy:
        readable_policy.append(str(Action(a)).replace('Action.', ''))
    readable_policy = np.array(readable_policy).reshape(
            int(np.sqrt(N_STATES)),
            int(np.sqrt(N_STATES)))

    plt.matshow(argmax_policy.reshape(
            int(np.sqrt(N_STATES)),
            int(np.sqrt(N_STATES))))
    for (i, j), z in np.ndenumerate(readable_policy):
        plt.text(i, j, readable_policy[j, i], ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()
