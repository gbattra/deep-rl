# Greg Attra
# 02/17/2022

'''
Domain representation of the rental car problem
'''

from os import stat
from typing import Tuple
import numpy as np
import math
import matplotlib.pyplot as plt

from dataclasses import dataclass

MAX_CARS_START: int = 25
MAX_CARS_END: int = 20
N_CAR_MOVES: int = 5
REQ_RWD: float = 10.
MOVE_RWD: float = -2.

@dataclass
class Site:
    sign: int
    req_dynamics: np.ndarray
    ret_dynamics: np.ndarray
    transitions: np.ndarray
    rewards: np.ndarray


def site_init(
    req_lambda: float,
    ret_lambda: float,
    sign: float) -> Site:
    '''
    Build a site given its config and the state/action space
    '''
    req_dynamics, ret_dynamics = site_build_dynamics(req_lambda, ret_lambda)
    transitions, rewards = site_compute_transitions(req_dynamics, ret_dynamics)
    return Site(sign, req_dynamics, ret_dynamics, transitions, rewards)


def poisson(n: float, lam: float) -> float:
    '''
    Compute the poisson probability of 'n' given a
    lambda of 'lam'
    '''
    return ((lam**n) / math.factorial(n)) * (np.e ** -lam)


def site_build_dynamics(
        req_lambda: float,
        ret_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Setup the dynamics for a given site
    '''
    req_dynamics = np.zeros((MAX_CARS_START + 1))
    ret_dynamics = np.zeros((MAX_CARS_START + 1))
    for s in range(MAX_CARS_START + 1):
        req_dynamics[s] = poisson(s, req_lambda)
        ret_dynamics[s] = poisson(s, ret_lambda)

    return req_dynamics, ret_dynamics


def site_compute_transitions(
        req_dynamics: np.ndarray,
        ret_dynamics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute the probabilities of each possible end state
    for each possible start state. Update the dynamics and reward
    mappings.
    '''
    # we can start the day with MAX_CARS + N_MOVES cars at a site
    transitions = np.zeros((MAX_CARS_START + 1, MAX_CARS_END + 1))
    rewards = np.zeros((MAX_CARS_START + 1, MAX_CARS_END + 1))
    for start in range(MAX_CARS_START + 1):
        if start == 0:
            continue

        rwd = 0
        rwd_counter = 0
        for end in range(MAX_CARS_END + 1):
            p = .0
            for n_req in range(MAX_CARS_END):
                delta = start - n_req
                if delta <= 0:
                    continue
                for n_ret in range(MAX_CARS_END):
                    # if the state after requests and returns doesn't match end state
                    # ignore this case / continue
                    if delta + n_ret != end:
                        continue

                    # add probability of receiving n_requests and n_returns to
                    # running probability
                    p += req_dynamics[n_req] * ret_dynamics[n_ret]
                    rwd += req_dynamics[n_req] * REQ_RWD
            rwd_counter += 1

            # store the computed joint probabilities in the transition matrix
            transitions[start, end] = p

            rewards[start, end] = rwd
    return transitions, rewards


def action_idx_to_value(a: int, site: Site) -> int:
    '''
    if sign = -1 and a < 0: add cars to this site; if a > 0, remove cars from site
    if sign = 1 and a < 0: remove cars from this stie; if a > 0, add cars to this site
    '''
    return (site.sign * (a - N_CAR_MOVES))


def site_next_start(s: int, a: int, site: Site) -> int:
    '''
    Compute the next start state (deterministic) for a
    given site
    '''
    a_val = action_idx_to_value(a, site)
    # clamp between 25 and 0
    return np.minimum(
                    MAX_CARS_START,
                    np.maximum(0, int(s + a_val)))


def site_reward(s: int, a: int, s_prime: int, site: Site) -> float:
    '''
    Get the expected reward for taking action a at state s for
    a given site
    '''
    start = site_next_start(s, a, site)
    a_val = action_idx_to_value(a, site)
    rwd = site.rewards[start, s_prime] + (-1. * np.abs(a_val))
    return rwd
    

def site_transition(s_prime: int, s: int, a: int, site: Site) -> float:
    '''
    Compute probability of ending day at state s_prime if taking
    action a at state s for a given site
    '''
    start = site_next_start(s, a, site)
    return site.transitions[start, s_prime]
