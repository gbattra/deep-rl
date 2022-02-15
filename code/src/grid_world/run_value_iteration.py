#!/usr/bin/env python

# Greg Attra
# 02/12/2022

# Driver class for running value iteration in the grid world problem

import numpy as np
import matplotlib.pyplot as plt

from lib.grid_world.grid_world import N_STATES, gw_dynamics, Action
from lib.grid_world.value_iteration import value_iteration


def main():
    dynamics = gw_dynamics()
    policy = value_iteration(dynamics)
    argmax_policy = np.argmax(policy, axis=1)

    readable_policy = []
    for a in argmax_policy:
        readable_policy.append(str(Action(a)).replace('Action.', ''))
    readable_policy = np.array(readable_policy)

    flipped_readable_policy = np.flip(
        readable_policy.reshape(
            int(np.sqrt(N_STATES)),
            int(np.sqrt(N_STATES))),
        axis=0)
    flipped_argmax_policy = np.flip(
        argmax_policy.reshape(
            int(np.sqrt(N_STATES)),
            int(np.sqrt(N_STATES))),
        axis=0)

    plt.matshow(flipped_argmax_policy)
    for (i, j), z in np.ndenumerate(flipped_readable_policy):
        plt.text(i, j, flipped_readable_policy[j, i], ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.show()


if __name__ == '__main__':
    main()
