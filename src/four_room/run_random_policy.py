#!/usr/bin/env python

# Greg Attra
# 01/22/2022

# Driver class for running a random policy agent in the four room problem

import numpy as np
import matplotlib.pyplot as plt

from lib.four_room.agent import Agent
from lib.four_room.policy import RandomPolicy
from lib.four_room.simulate import MapCell, Simulation
from lib.four_room.map import map_instantiate


def main():
    mapp = map_instantiate((11, 11), (10, 10))
    sim = Simulation(mapp)
    init_state = sim.reset()
    policy = RandomPolicy()
    agent = Agent(init_state, policy, sim)

    n_trials = 10
    n_steps = 10**4

    cu_trial_rewards = np.zeros((n_trials, n_steps))
    for t in range(n_trials):
        cu_reward = 0
        for s in range(n_steps):
            transition = agent.step()
            agent.update(transition)

            cu_reward += transition.reward
            cu_trial_rewards[t, s] = cu_reward

    for t in range(n_trials):
        plt.plot(cu_trial_rewards[t, :], ':')
    
    mean = np.mean(cu_trial_rewards, 0)
    plt.plot(mean, 'k')

    plt.show()
    
    

if __name__ == '__main__':
    main()
