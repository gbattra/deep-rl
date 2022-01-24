#!/usr/bin/env python

# Greg Attra
# 01/22/2022

# Driver class for evaluating a random policy, a fixed action policy
# and a follow path policy in the four room problem

from typing import List
import numpy as np
import matplotlib.pyplot as plt

from lib.four_room.agent import Agent
from lib.four_room.policy import SANSPolicy, RandomPolicy
from lib.four_room.simulate import Simulation
from lib.four_room.map import map_instantiate


def step_agent(
        agent: Agent,
        rewards: np.ndarray,
        t: int, s: int) -> None:
    transition = agent.step()
    agent.update(transition)

    if s == 0:
        rewards[t, s] = transition.reward
    else:
        rewards[t, s] = rewards[t, s - 1] + transition.reward


def plot_rewards(agent_rewards: List[np.ndarray], n_trials: int) -> None:
    for _, rewards, label in agent_rewards:
        for t in range(n_trials):
            plt.plot(rewards[t, :], ':')
        
        mean = np.mean(rewards, 0)
        plt.plot(mean, label=label)

    plt.xlabel('Steps')
    plt.ylabel('Cummulative Reward')
    plt.legend()
    plt.show()


def main() -> None:
    mapp = map_instantiate((11, 11), (10, 10))
    sim = Simulation(mapp)
    init_state = sim.reset()
    
    random_policy = RandomPolicy()
    random_agent = Agent(init_state, random_policy, sim)

    sans_policy = SANSPolicy(500)
    sans_500_agent = Agent(init_state, sans_policy, sim)

    sans_policy = SANSPolicy(10)
    sans_10_agent = Agent(init_state, sans_policy, sim)

    n_trials = 10
    n_steps = 10**4

    random_policy_rewards = np.zeros((n_trials, n_steps))
    sans_500_policy_rewards  = np.zeros((n_trials, n_steps))
    sans_10_policy_rewards  = np.zeros((n_trials, n_steps))

    agents_rewards = [
        (random_agent, random_policy_rewards, 'Random Policy'),
        (sans_500_agent, sans_500_policy_rewards, 'SANS-500 Policy'),
        (sans_10_agent, sans_10_policy_rewards, 'SANS-10 Policy')
    ]
    for t in range(n_trials):
        for s in range(n_steps):
            for agent, rewards, _ in agents_rewards:
                step_agent(agent, rewards, t, s)

    plot_rewards(agents_rewards, n_trials)
    

if __name__ == '__main__':
    main()
