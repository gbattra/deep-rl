# Greg Attra
# 02/17/2022

'''
Domain representation of the rental car problem
'''

from typing import Tuple
import numpy as np
import math

from dataclasses import dataclass

MAX_CARS: int = 20
N_CAR_MOVES: int = 5
SITE_STATE_SPACE: int = MAX_CARS + N_CAR_MOVES
REQ_RWD: int = 10
MOVE_RWD: int = -2

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


def site_build_dynamics(req_lambda: float, ret_lambda: float) \
    -> Tuple[np.ndarray, np.ndarray]:
    '''
    Setup the dynamics for a given site
    '''
    req_dynamics = np.zeros((SITE_STATE_SPACE))
    ret_dynamics = np.zeros((SITE_STATE_SPACE))
    for s in range(SITE_STATE_SPACE):
        req_dynamics[s] = poisson(s, req_lambda)
        ret_dynamics[s] = poisson(s, ret_lambda)

    return req_dynamics, ret_dynamics


def site_compute_transitions(
        req_dynamics: np.ndarray,
        ret_dynamics: np.ndarray) -> None:
    '''
    Compute the probabilities of each possible end state
    for each possible start state. Update the dynamics and reward
    mappings.
    '''
    # we can start the day with MAX_CARS + N_MOVES cars at a site
    transitions = np.ndarray((SITE_STATE_SPACE, SITE_STATE_SPACE))
    for start in range(SITE_STATE_SPACE):
        # if start state has zero cars, out of bisiness
        if start == 0:
            return .0
        for end in range(MAX_CARS):
            # only one site can have > 20 cars
            if start > MAX_CARS and end > MAX_CARS:
                return .0
            p = .0
            # MAX_CARS + N_MOVES = 25 (the max number of cars at a lot after hours)
            for n_req in range(SITE_STATE_SPACE):
                for n_ret in range(SITE_STATE_SPACE):
                    delta = start - n_req
                    # if we receive more requests than cars in the lot,
                    # business is lost
                    if delta <= 0:
                        continue
                    # if the state after requests and returns doesn't match end state
                    # ignore this case / continue
                    if delta + n_ret != end:
                        continue

                    # add probability of receiving n_requests and n_returns to
                    # running probability
                    p += req_dynamics[n_req] * ret_dynamics[n_ret]

            # store the computed joint probabilities in the transition matrix
            transitions[start, end] = p

        # compute rewards
        rewards = np.zeros((SITE_STATE_SPACE))
        rwd = 0
        n_cars = start
        while n_cars > 0:
            # rwd += the probability of n_cars requests in a day X the reward
            # for renting n_cars
            rwd += req_dynamics[n_cars] * (n_cars * REQ_RWD)
        rewards[start] = rwd
        return transitions, rewards


def site_next_start(s: int, a: int, site: Site) -> int:
    '''
    Compute the next start state (deterministic) for a
    given site
    '''
    # if sign = -1 and a < 0: add cars to this site; if a > 0, remove cars from site
    # if sign = 1 and a < 0: remove cars from this stie; if a > 0, add cars to this site
    a -= N_CAR_MOVES
    return s + (site.sign * a)


def site_rewards(s: int, a: int, site: Site) -> float:
    '''
    Get the expected reward for taking action a at state s for
    a given site
    '''
    s_prime = site_next_start(s, a, site)
    return site.rewards[s_prime]
    

def site_transition(s_prime: int, s: int, a: int, site: Site) -> float:
    '''
    Compute probability of ending day at state s_prime if taking
    action a at state s for a given site
    '''
    start = site_next_start(s, a, site)
    return site.transitions[start, s_prime]
