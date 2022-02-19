# Greg Attra
# 02/17/2022

'''
Domain representation of the rental car problem
'''

import numpy as np
import math

from dataclasses import dataclass

MAX_CARS: int = 20
N_CAR_MOVES: int = 5
N_ACIONS: int = 11
REQ_RWD: int = 10
MOVE_RWD: int = -2

@dataclass
class Site:
    sign: int
    req_lambda: float
    ret_lambda: float
    req_dynamics: np.ndarray
    ret_dynamics: np.ndarray
    transitions: np.ndarray
    rewards: np.ndarray


def poisson(n: float, lam: float) -> float:
    '''
    Compute the poisson probability of 'n' given a
    lambda of 'lam'
    '''
    return ((lam**n) / math.factorial(n)) * (np.e ** -lam)


def build_site_dynamics(site: Site) -> None:
    '''
    Setup the dynamics for a given site
    '''
    for n in range(MAX_CARS + N_CAR_MOVES):
        site.req_dynamics[n] = poisson(n, site.req_lambda)
        site.ret_dynamics[n] = poisson(n, site.ret_lambda)


def compute_transitions(site: Site) -> None:
    '''
    Compute the probabilities of each possible end state
    for each possible start state. Update the dynamics and reward
    mappings.
    '''
    # we can start the day with MAX_CARS + N_MOVES cars at a site
    for start in range(MAX_CARS + N_CAR_MOVES):
        # if start state has zero cars, out of bisiness
        if start == 0:
            return .0
        for end in range(MAX_CARS):
            # only one site can have > 20 cars
            if start > MAX_CARS and end > MAX_CARS:
                return .0
            p = .0
            # MAX_CARS + N_MOVES = 25 (the max number of cars at a lot after hours)
            for n_req in range(MAX_CARS + N_CAR_MOVES):
                for n_ret in range(MAX_CARS + N_CAR_MOVES):
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
                    p += site.req_dynamics[n_req] * site.ret_dynamics[n_ret]

            # store the computed joint probabilities in the transition matrix
            site.transitions[start, end] = p

        # compute rewards
        rwd = 0
        n_cars = start
        while n_cars > 0:
            # rwd += the probability of n_cars requests in a day X the reward
            # for renting n_cars
            rwd += site.req_dynamics[n_cars] * (n_cars * REQ_RWD)
        site.rewards[start] = rwd


def next_start(s: int, a: int, site: Site) -> int:
    '''
    Compute the next start state (deterministic) for a
    given site
    '''
    # if sign = -1 and a < 0: add cars to this site; if a > 0, remove cars from site
    # if sign = 1 and a < 0: remove cars from this stie; if a > 0, add cars to this site
    a -= N_CAR_MOVES
    return s + (site.sign * a)


def rewards(s: int, a: int, site: Site) -> float:
    '''
    Get the expected reward for taking action a at state s for
    a given site
    '''
    s_prime = next_start(s, a, site)
    return site.rewards[s_prime]
    

def transition(s_prime: int, s: int, a: int, site: Site) -> float:
    '''
    Compute probability of ending day at state s_prime if taking
    action a at state s for a given site
    '''
    start = next_start(s, a, site)
    return site.transitions[start, s_prime]
