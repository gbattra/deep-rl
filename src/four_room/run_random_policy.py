#!/usr/bin/env python

# Greg Attra
# 01/22/2022

# Driver class for running a random policy agent in the four room problem


from lib.four_room.agent import Agent
from lib.four_room.policy import RandomPolicy
from lib.four_room.simulate import Simulation
from lib.four_room.map import map_instantiate


def main():
    mapp = map_instantiate((11, 11), (10, 10))
    sim = Simulation(mapp)
    init_state = sim.reset()
    policy = RandomPolicy()
    agent = Agent(init_state, policy, sim)

    n_trials = 10
    n_steps = 10^4

    all_cu_trial_rewards = []
    for t in range(n_trials):
        cu_trial_rewards = []
        for s in range(n_steps):
            transition = agent.step()
            cu_trial_rewards.append(transition.reward)
            agent.update(transition)
        all_cu_trial_rewards.append(cu_trial_rewards)


if __name__ == '__main__':
    main()
