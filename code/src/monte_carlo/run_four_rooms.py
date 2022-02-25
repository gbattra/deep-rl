# Greg Attra
# 02/25/2022

'''
Exectuable for running four rooms MC control
'''

import matplotlib.pyplot as plt

from lib.monte_carlo.env import FourRoomsEnv
from lib.monte_carlo.algorithms import on_policy_mc_control_epsilon_soft

def main():
    env = FourRoomsEnv()
    returns = on_policy_mc_control_epsilon_soft(
        env,
        int(10e4),
        0.99,
        .1)

    plt.plot(returns)
    plt.show()

if __name__ == '__main__':
    main()
