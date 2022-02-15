# Greg Attra
# 02/14/2022

'''
Implementation of value iteration
'''

import numpy as np

from lib.grid_world.grid_world import N_STATES, Action


MAX_ITER = 10000
DISCOUNT_FACTOR = 0.9
THETA = 10e-3


def value_iteration(dynamics: np.ndarray) -> np.ndarray:
    '''
    Iterate over the state space to estimate the value of each
    state using value iteration.

    Output a deterministic policy approximating the optimal policy.
    '''
    V = np.zeros((N_STATES))
    policy = np.zeros((N_STATES, len(Action)))

    # iter counter to avoid inf loop
    i = 0
    # update value estimates for all states until convergence
    while True and i < MAX_ITER:
        # the change in the values after this iteration
        delta = 0
        for s in range(N_STATES):
            # current value for state s
            old_v = V[s]
            # the value for state s to compute
            v_sum = 0
            a_values = []
            for a in range(len(Action)):
                # prob of choosing action a at state s
                s_sum = 0
                for s_prime in range(N_STATES):
                    # rwd and prob of entering s_prime from state s taking aciton a
                    p, r = dynamics[s, a, s_prime]
                    # value at s_prime
                    v = V[s_prime]
                    # add value to running sum
                    s_sum += p * (r + DISCOUNT_FACTOR * v)
                a_values.append(s_sum)
            # update value function for state s
            a_max = np.argmax(a_values)
            V[s] = a_values[a_max]

            # update policy to use max action
            policy[s, :] = 0
            policy[s, a_max] = 1
            
            # update delta in value for state s
            delta = np.maximum(delta, np.abs(V[s] - old_v))
        # check if converged
        if delta <= THETA:
            break
        
        i += 1

    return policy
