# Greg Attra
# 02/18/2022

'''
Functions for representing the dynamics of the rental
car business
'''

import numpy as np

from typing import Tuple
from lib.rental_car.site import Site, N_CAR_MOVES, MAX_CARS, rewards, transition


N_STATES = (N_CAR_MOVES + MAX_CARS) ** 2
ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def state_coords_to_id(x: int, y: int) -> int:
    '''
    Convert the x, y coords of the state to the
    int id of the state
    '''
    return int((y * np.sqrt(N_STATES)) + x)


def state_id_to_coords(s: int) -> Tuple[int, int]:
    '''
    Convert the id of the state to the x, y coord
    of the state cell
    '''
    y = s / np.sqrt(N_STATES)
    x = s % np.sqrt(N_STATES)

    return int(x), int(y)


def business_dynamics(site_a: Site, site_b: Site) -> np.ndarray:
    '''
    Build the complete dynamics matrix for using sites
    '''
    dynamics = np.zeros((N_STATES, len(ACTIONS), N_STATES, 2))
    for s in range(N_STATES):
        start_a, start_b = state_id_to_coords(s)
        for a in ACTIONS:
            for s_prime in range(N_STATES):
                end_a, end_b = state_id_to_coords(s_prime)
                # expected rewards taking action a at site s for each site
                r_a = rewards(s, a, site_a)
                r_b = rewards(s, a, site_b)
                # transition probability for each site
                p_a = transition(end_a, start_a, a, site_a)
                p_b = transition(end_b, start_b, a, site_b)
                dynamics[s, a, s_prime, :] = p_a * p_b, r_a + r_b

    return dynamics
