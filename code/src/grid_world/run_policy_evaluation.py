#!/usr/bin/env python

# Greg Attra
# 02/12/2022

# Driver class for running policy evaluation in the grid world problem

import numpy as np
import matplotlib.pyplot as plt

from lib.grid_world.grid_world import N_STATES, gw_dynamics, Action
from lib.grid_world.policy_iteration import policy_evaluation


def main():
    policy = np.ones((N_STATES, len(Action))) / float(len(Action))
    dynamics = gw_dynamics()
    values = policy_evaluation(policy, dynamics)
    print(values)


if __name__ == '__main__':
    main()
