# Greg Attra
# 02/18/2022

'''
Functions for representing the dynamics of the rental
car business
'''

import numpy as np

from typing import Tuple
from lib.rental_car.site import (
    MAX_CARS_END, Site, site_next_start, site_reward, site_transition)


N_STATES = (MAX_CARS_END + 1) ** 2
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
    dynamics = np.zeros((
        N_STATES,
        len(ACTIONS),
        N_STATES,
        2))
    for s in range(N_STATES):
        print(f'Computing dynamics... {s}/{N_STATES}', end='\r')
        start_a, start_b = state_id_to_coords(s)
        for a in ACTIONS:
            for s_prime in range(N_STATES):
                end_a, end_b = state_id_to_coords(s_prime)
                # transition probability for each site
                p_a = site_transition(end_a, start_a, a, site_a)
                p_b = site_transition(end_b, start_b, a, site_b)

                # expected rewards taking action a at site s for each site
                r_a = site_reward(start_a, a, end_a, site_a)
                r_b = site_reward(start_b, a, end_b, site_b)

                # print(f'[s_a: {start_a}, s_b: {start_b}, a: {a}] -> [e_a: {end_a}, e_b: {end_b}]')
                # print(f'p_a: {p_a}, p_b: {p_b}, r_a: {r_a}, r_b: {r_b}')
                # print('-----------')

                p = p_a * p_b
                rwd = r_a + r_b

                dynamics[s, a, s_prime, :] = p, rwd
    return dynamics
