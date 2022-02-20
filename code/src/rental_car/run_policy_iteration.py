# Greg Attra
# 02/18/2022

'''
Run policy iteration on the simple version of the rental car problem
'''

import numpy as np
import matplotlib.pyplot as plt

from lib.rental_car.business import N_STATES, ACTIONS, business_dynamics
from lib.rental_car.policy_iteration import policy_iteration
from lib.rental_car.site import site_init

SITE_A_CONFIG: dict = {
    'req_lambda': 3.,
    'ret_lambda': 3.,
    'sign': 1.
}

SITE_B_CONFIG: dict = {
    'req_lambda': 4.,
    'ret_lambda': 2.,
    'sign': -1.
}

def main():
    # initialize policy
    site_a = site_init(
        SITE_A_CONFIG['req_lambda'],
        SITE_A_CONFIG['ret_lambda'],
        SITE_A_CONFIG['sign'])
    site_b = site_init(
        SITE_B_CONFIG['req_lambda'],
        SITE_B_CONFIG['ret_lambda'],
        SITE_B_CONFIG['sign'])
    biz_dynamics = business_dynamics(site_a, site_b)

    policy = np.zeros((N_STATES, len(ACTIONS)))
    policy[:, 5] = 1.
    policy = policy_iteration(policy, biz_dynamics, N_STATES, len(ACTIONS))

    argmax_policy = np.argmax(policy, axis=1).reshape(
                int(np.sqrt(N_STATES)),
                int(np.sqrt(N_STATES)))
    plt.matshow(argmax_policy)

    plt.gca().invert_yaxis()
    plt.show()
    print(policy)


if __name__ == '__main__':
    main()
