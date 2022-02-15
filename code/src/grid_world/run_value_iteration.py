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

    print('Value Function:')
    print(np.flip(
            readable_policy.reshape(
                int(np.sqrt(N_STATES)),
                int(np.sqrt(N_STATES))),
            axis=0))


if __name__ == '__main__':
    main()