# Greg Attra
# 02/17/2022

'''
Domain representation of the rental car problem
'''

from typing import Tuple
import numpy as np

from scipy.stats import poisson
from dataclasses import dataclass

MAX_CARS_START: int = 25
MAX_CARS_END: int = 20
N_CAR_MOVES: int = 5
DYNAMICS_DIST_SIZE: int = 11
REQ_RWD: float = 10.
MOVE_RWD: float = -2.


@dataclass
class Site:
    sign: int
    req_dynamics: np.ndarray
    ret_dynamics: np.ndarray
    transitions: np.ndarray
    rewards: np.ndarray
    has_assistant: bool
    overflow_penalty: bool


def site_init(
    req_lambda: float,
    ret_lambda: float,
    sign: float,
    has_assistant: bool = False,
    overflow_penalty: bool = False) -> Site:
    '''
    Build a site given its config and the state/action space
    '''
    req_dynamics, ret_dynamics = site_build_dynamics(req_lambda, ret_lambda)
    transitions, rewards = site_compute_transitions(req_lambda, ret_lambda)
    return Site(sign, req_dynamics, ret_dynamics, transitions, rewards, has_assistant, overflow_penalty)


def poisson_dist(n: float, lam: float) -> float:
    '''
    Compute the poisson probability of 'n' given a
    lambda of 'lam'
    '''
    # return ((lam**n) / math.factorial(n)) * (np.e ** -lam)
    return poisson.pmf(k=n, mu=lam)


def site_build_dynamics(
        req_lambda: float,
        ret_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Setup the dynamics for a given site
    '''
    req_dynamics = np.zeros((DYNAMICS_DIST_SIZE + 1))
    ret_dynamics = np.zeros((DYNAMICS_DIST_SIZE + 1))
    for s in range(DYNAMICS_DIST_SIZE + 1):
        req_dynamics[s] = poisson_dist(s, req_lambda)
        ret_dynamics[s] = poisson_dist(s, ret_lambda)

    return req_dynamics, ret_dynamics


def req_dist(state: int, req_lambda: float) -> np.ndarray:
    '''
    Compute the likelihood of getting n = 0 -> 25 requests
    given a start state
    '''
    k = np.arange(0, -26, step=-1) + state
    return poisson.pmf(k=k, mu=req_lambda)


def ret_dist(ret_lambda: float) -> np.ndarray:
    '''
    Compute likelihood of getting n = 0 -> 25 returns
    in a day at start state
    Inspired by code
    '''
    k = np.arange(0, 26, step=1)
    return poisson.pmf(k=k, mu=ret_lambda)


def site_compute_transitions(
        req_lambda: np.ndarray,
        ret_lambda: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute the probabilities of each possible end state
    for each possible start state. Update the dynamics and reward
    mappings.
    '''
    # we can start the day with MAX_CARS + N_MOVES cars at a site
    transitions = np.zeros((MAX_CARS_START + 1, MAX_CARS_END + 1))
    rewards = np.zeros((MAX_CARS_START + 1))
    for start in range(MAX_CARS_START + 1):
        req_dynamics = req_dist(start, req_lambda)
        for n_req in range(MAX_CARS_START):
            p_req = req_dynamics[n_req]

            ret_dynamics = ret_dist(ret_lambda)
            for n_ret in range(MAX_CARS_END):
                end = start - n_req + n_ret
                end = np.minimum(np.maximum(end, 0), MAX_CARS_END)
                p_ret = ret_dynamics[n_ret]
                transitions[start, end] += p_req * p_ret
        
        req_probs = req_dist(start, req_lambda)
        rwd = np.average([REQ_RWD * n for n in range(0, 26)], axis=0, weights=req_probs)
        rewards[start] = rwd
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
    return int(s + a_val)


def site_reward(s: int, a: int, site: Site) -> float:
    '''
    Get the expected reward for taking action a at state s for
    a given site
    '''
    start = site_next_start(s, a, site)
    a_val = action_idx_to_value(a, site)
    if start < 0:
        return -1. * np.abs(a_val)
    
    overflow_penalty = 4 if site.overflow_penalty and start > 10 else 0
    a_val_prime = max(0, a_val - 1) if site.has_assistant else a_val
    rwd = site.rewards[start] + (-1. * np.abs(a_val_prime)) - overflow_penalty
    return rwd
    

def site_transition(s_prime: int, s: int, a: int, site: Site) -> float:
    '''
    Compute probability of ending day at state s_prime if taking
    action a at state s for a given site
    '''
    start = site_next_start(s, a, site)
    if start < 0:
        return .0
    return site.transitions[start, s_prime]
