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
        dynamics: np.ndarray,
        n_states: int,
        n_actions: int) -> np.ndarray:
    '''
    Evaluate (estimate the value function) for a given policy
    and dynamics/rewards function
    '''
    # instantiate value function
    V = np.zeros((n_states))

    # iter counter to avoid inf loop
    i = 0
    # update values for all states until convergence
    while True and i < MAX_ITER:
        # the change in the values after this iteration
        delta = 0
        # re-estimate value for each state
        for s in range(n_states):
            print(f'Policy Eval: {i}, state: {s}/{n_states}', end='\r')
            # current value for state s
            old_v = V[s]
            # the value for state s to compute
            v_sum = 0
            for a in range(n_actions):
                # prob of choosing action a at state s
                a_prob = policy[s, a]
                s_sum = 0
                for s_prime in range(n_states):
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
        
        i += 1
    
    return V


def policy_iteration(
    dynamics: np.ndarray,
    n_states: int,
    n_actions: int) -> np.ndarray:
    '''
    Estimate the optimal policy using policy iteration
    '''
    # instantiate arbitrary policy
    policy = np.zeros((n_states, n_actions))

    i = 0
    while True and i < MAX_ITER:
        # estimate value function for policy
        V = policy_evaluation(policy, dynamics, n_states, n_actions)

        stable = True
        for s in range(n_states):
            print(f'Policy Iter: {i}, state: {s}/{n_states}', end='\r')
            a_values = []
            for a in range(n_actions):
                # prob of choosing action a at state s
                s_sum = 0
                for s_prime in range(n_states):
                    # rwd and prob of entering s_prime from state s taking aciton a
                    p, r = dynamics[s, a, s_prime]
                    # value at s_prime
                    v = V[s_prime]
                    # add value to running sum
                    s_sum += p * (r + DISCOUNT_FACTOR * v)
                a_values.append(s_sum)
            # store the current policy action for comparison
            old_a = np.argmax(policy[s])
            # get action with max value
            a_max = np.argmax(a_values)
            # update policy to use max action
            policy[s, :] = 0
            policy[s, a_max] = 1

            # if new action is not same as old action, and new action value
            # is greater than old action value, policy not stable
            if a_max != old_a and a_values[a_max] > a_values[old_a]:
                stable = False
        
        if stable:
            break
        
        i += 1

    return policy
