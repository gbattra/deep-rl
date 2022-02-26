# Greg Attra
# 02/24/2022

'''
Executable for running MC eval on racetrack problem
'''

import matplotlib.pyplot as plt
import numpy as np

from lib.monte_carlo.algorithms import on_policy_mc_control_epsilon_soft
from lib.monte_carlo.racetracks import Racetrack, track0

N_EPISODES = int(10**4)
GOAL_RWD = 1.0
N_TRIALS = 2
GAMMA = .99

def main():
    track = track0()
    env = Racetrack(track)
    returns = on_policy_mc_control_epsilon_soft(
        env,
        N_EPISODES,
        GAMMA,
        0.1)
    
    plt.plot(returns)
    plt.show()

if __name__ == '__main__':
    main()
