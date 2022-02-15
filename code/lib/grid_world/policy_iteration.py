# Greg Attra
# 02/12/2021

'''
Optimizer class for evaluating and improving a policy
'''

import numpy as np

from lib.grid_world.grid_world import N_STATES, Action


MAX_ITER = 10000
DISCOUNT_FACTOR = 0.9
THETA = 10e-3


def policy_evaluation(
        policy: np.ndarray,
        dynamics: np.ndarray):
    '''
    Evaluate (estimate the value function) for a given policy
    and dynamics/rewards function
    '''
    V = np.zeros((N_STATES))

    # iter counter to avoid inf loop
    i = 0
    # update values for all states until convergence
    while True and i < MAX_ITER:
        # the threshold to use to end 
        delta = 0
        # re-estimate value for each state
        for s in range(N_STATES):
            # current value for state s
            old_v = V[s]
            # the value for state s to compute
            v_sum = 0
            for a in range(len(Action)):
                # prob of choosing action a at state s
                a_prob = policy[s, a]
                s_sum = 0
                for s_prime in range(N_STATES):
                    # rwd and prob of entering s_prime from state s taking aciton a
                    p, r = dynamics[s, a, s_prime]
                    # value at s_prime
                    v = V[s_prime]
                    # add value to running sum
                    s_sum += p * (r + DISCOUNT_FACTOR * v)
                v_sum += a_prob * s_sum
            # update value function for state s
            V[s] = v_sum
            # update delta in value for state s
            delta = np.maximum(delta, np.abs(V[s] - old_v))
        # check if converged
        if delta <= THETA:
            break

    return V
