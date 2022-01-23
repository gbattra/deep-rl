# Greg Attra
# 01/23/2022

# Class for visualizing the simulation and agent state

import copy
import numpy as np

from lib.four_room.simulate import Simulation
from lib.four_room.agent import Agent


def visualize_scenario(agent: Agent, simulation: Simulation):
    """
    Print the map and agent state
    """
    x, y = agent.state
    mapp = copy.deepcopy(simulation.map)
    mapp[y, x] = 8  # number representing agent in matrix
    mapp = np.flip(mapp, axis=0)
    print(mapp)
